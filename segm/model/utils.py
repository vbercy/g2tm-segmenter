# MIT License

# Copyright (c) 2021 Robin Strudel
# Copyright (c) INRIA

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import math

import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

import segm.utils.torch as ptu


def init_weights(m):
    """ Randomly initialize weights for some layers.
    """
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    """ Rescale the grid of position embeddings when loading from state_dict.
    Adapted from
    https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    posemb_tok, posemb_grid = (
        posemb[:, :num_extra_tokens],
        posemb[0, num_extra_tokens:],
    )
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = (
        posemb_grid.reshape(1, gs_old_h, gs_old_w, -1)
                   .permute(0, 3, 1, 2)
    )
    posemb_grid = F.interpolate(
        posemb_grid, size=(gs_h, gs_w), mode="bilinear"
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)  # pylint: disable=E1101
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """Convert patch embedding weight from manual patchify + linear proj
    to conv.
    """
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    num_extra_tokens = 1 + ("dist_token" in state_dict.keys())
    patch_size = model.patch_size
    image_size = model.patch_embed.image_size
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from
            # pretrained weights
            v = resize_pos_embed(
                v,
                None,
                (image_size[0] // patch_size, image_size[1] // patch_size),
                num_extra_tokens,
            )
        out_dict[k] = v
    return out_dict


def padding(im, patch_size, fill_value=0):
    """ Make the image sizes divisible by patch_size.
    """
    h, w = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if h % patch_size > 0:
        pad_h = patch_size - (h % patch_size)
    if w % patch_size > 0:
        pad_w = patch_size - (w % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)  # pylint: disable=E1102
    return im_padded


def unpadding(y, target_size):
    """ Crop predictions on extra pixels coming from padding.
    """
    h, w = target_size
    h_pad, w_pad = y.size(2), y.size(3)
    extra_h = h_pad - h
    extra_w = w_pad - w
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


def resize(im, smaller_size):
    """ Resize image to a smaller size using bilinear interpolation.
    """
    h, w = im.shape[2:]
    if h < w:
        ratio = w / h
        h_res, w_res = smaller_size, ratio * smaller_size
    else:
        ratio = h / w
        h_res, w_res = ratio * smaller_size, smaller_size
    if min(h, w) < smaller_size:
        im_res = F.interpolate(im, (int(h_res), int(w_res)), mode="bilinear")
    else:
        im_res = im
    return im_res


def sliding_window(im, flip, window_size, window_stride):
    """ Slice the image into fixed-size windows for inference.
    """
    _, _, h, w = im.shape
    ws = window_size

    windows = {"crop": [], "anchors": []}
    h_anchors = torch.arange(0, h, window_stride)  # pylint: disable=E1101
    w_anchors = torch.arange(0, w, window_stride)  # pylint: disable=E1101
    h_anchors = [ha.item() for ha in h_anchors if ha < h - ws] + [h - ws]
    w_anchors = [wa.item() for wa in w_anchors if wa < w - ws] + [w - ws]
    for ha in h_anchors:
        for wa in w_anchors:
            window = im[:, :, ha:ha + ws, wa:wa + ws]
            windows["crop"].append(window)
            windows["anchors"].append((ha, wa))
    windows["flip"] = flip
    windows["shape"] = (h, w)
    return windows


def merge_windows(windows, window_size, ori_shape):
    """ Merge predictions of all windows into a single segmentation map.
    """
    ws = window_size
    im_windows = windows["seg_maps"]
    anchors = windows["anchors"]
    c = im_windows[0].shape[0]
    h, w = windows["shape"]
    flip = windows["flip"]

    logit = torch.zeros((c, h, w), device=im_windows.device)  # pylint: disable=E1101
    count = torch.zeros((1, h, w), device=im_windows.device)  # pylint: disable=E1101
    for window, (ha, wa) in zip(im_windows, anchors):
        logit[:, ha:ha + ws, wa:wa + ws] += window
        count[:, ha:ha + ws, wa:wa + ws] += 1
    logit = logit / count
    logit = F.interpolate(
        logit.unsqueeze(0),
        ori_shape,
        mode="bilinear",
    )[0]
    if flip:
        logit = torch.flip(logit, (2,))  # pylint: disable=E1101
    result = F.softmax(logit, 0)
    return result


def inference(
    model,
    ims,
    ims_metas,
    ori_shape,
    window_size,
    window_stride,
    batch_size,
):
    """ Run inference on several images.
    """
    c = model.n_cls
    seg_map = torch.zeros((c, ori_shape[0], ori_shape[1]), device=ptu.device)  # pylint: disable=E1101
    for im, im_metas in zip(ims, ims_metas):
        im = im.to(ptu.device)
        im = resize(im, window_size)
        flip = False  # im_metas["flip"]
        windows = sliding_window(im, flip, window_size, window_stride)
        crops = torch.stack(windows.pop("crop"))[:, 0]  # pylint: disable=E1101
        b = len(crops)
        wb = batch_size
        seg_maps = torch.zeros((b, c, window_size, window_size),  # pylint: disable=E1101
                               device=im.device)
        with torch.no_grad():
            for i in range(0, b, wb):
                seg_maps[i:i + wb] = model.forward(crops[i:i + wb])
        windows["seg_maps"] = seg_maps
        im_seg_map = merge_windows(windows, window_size, ori_shape)
        seg_map += im_seg_map
    seg_map /= len(ims)
    return seg_map


def num_params(model):
    """ Compute total number of parameters of all layers in the model.
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([torch.prod(torch.tensor(p.size()))  # pylint: disable=E1101
                    for p in model_parameters])
    return n_params.item()

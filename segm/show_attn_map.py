# Copyright © 2025 Commissariat à l'Energie Atomique et aux Energies Alternatives (CEA)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications based on code from Robin Strudel et al. (Segmenter)

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


from pathlib import Path

import click
import einops
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from segm.data.utils import STATS
from segm.model.decoder import MaskTransformer
from segm.model.factory import load_model
from segm.model.utils import inference
import segm.utils.torch as ptu

import g2tm


def save_attention_map(attention_map: torch.Tensor, img_vis: Image.Image,
                       dir_path: Path, file_name: str, cmap: str):
    """Save an overlay of attention map on the image as PNG file.

    This function saves a PNG file containing an attention map (per patch)
    overlaying the original image. The attention scores are smoothed with a
    gaussian filter for prettier visualization.

    Args:
        attention_map (torch.Tensor): Attention map.
        img_vis (PIL.Image.Image): Original image.
        dir_path (Path): Path to directory where PGN file is saved.
        file_name (str): Name of the saved file.
        cmap (str): Color map argument.
    """
    file_path = dir_path / f"{file_name}.png"
    file_path_atten_overly = dir_path / f"{file_name}_overlay.png"
    attention_map_ = gaussian_filter(attention_map, sigma=10)

    plt.imsave(fname=str(file_path), arr=attention_map_,
               format="png", cmap=cmap)

    # overlay image
    attention_weights_normalized = (
        (attention_map - np.min(attention_map)) /
        (np.max(attention_map) - np.min(attention_map))
    )
    attention_weights_normalized = gaussian_filter(
        attention_weights_normalized, sigma=10
    )

    attention_map = (
        np.array(img_vis) * 0.6 +
        plt.get_cmap('jet')(attention_weights_normalized)[:, :, :3] * 255 * 0.4
    ).astype(np.uint8)
    plt.imsave(fname=str(file_path_atten_overly), arr=attention_map,
               format="png", cmap=cmap)
    print(f"{file_path} saved.")


@click.command()
@click.argument("model-path", type=str)
@click.argument("image-path", type=str)
@click.argument("output-dir", type=str)
@click.argument("cmap-file", type=str)
@click.option("--layer-id", default=0, type=int)
@click.option("--x-patch", default=0, type=int)
@click.option("--y-patch", default=0, type=int)
@click.option("--cls/--patch", default=False, is_flag=True)
@click.option("--enc/--dec", default=True, is_flag=True)
@click.option("--cmap", default="viridis", type=str)
@click.option("--patch-type", default="pure", type=str)
@click.option("--selected-layer", default=1, type=int)
@click.option("--threshold", default=0.88, type=float)
@click.option("--prop-attn/--no-prop-attn", default=False, is_flag=True)
@click.option("--iprop-attn/--no-iprop-attn", default=False, is_flag=True)
def visualize(model_path, image_path, output_dir, cmap_file,
              layer_id, x_patch, y_patch, cls, enc, cmap,
              patch_type, selected_layer, threshold, prop_attn, iprop_attn):
    """Attention, token and prediction visualizations of the model.

    This function creates several attention, token and predicition
    visualizations of a model at a specific layer for a specified image.
    It supports visualizations for both models with and without token
    reduction.
    1 - Attention map visualizations: for each head, at one layer
        a - Raw attention map
        b - Attention map overlaying the original image
    2 - Token visualizations: at one layer
        a - Token grid overlaying the original image with (merged) patches
            detoured.
        b - Token grid showing fusions with smoothed colors in each
            merged group of patches.
        c - Original ViT grid.
    3 - Segmentation visualizations
        a - Predicted segmentation map.
        b - Groundtruth segmentation map.
    4 - Patch visualization: if a specific patch is selected.
    For token visualization, if the model uses G2TM and if the token sequence
    has been reduced, 2.a. and 2.b. images are saved, otherwise 2.c.

    Args:
        model_path (str): Path to PyTorch model.
        image_path (str): Path to an image file.
        output_dir (str): Path to the directory where visualizations are saved.
        cmap_file (str): Path to the dataset colormap file.
        layer_id (int): Layer to visualize (0-based)
        x_patch (int): X-coordinate of the selected patch (optional).
        y_patch (int): Y-coordinate of the selected patch (optional).
        cls (bool): Whether to compute attention from the [CLS] token (True)
        or any selected patch (False).
        enc (bool): Whether the visualizations are made in the encoder (True)
        or in the decoder (False).
        cmap (str): Colormap used for attention visualizations.
        patch_type (str): Token reduction method (pure => no reduction).
        selected_layer (int): Layer to apply token reduction (1-based).
        threshold (float): Threshold parameter for G2TM.
        prop_attn (bool): Whether to apply Proportional Attention.
        iprop_attn (bool): Whether to apply Inverse Proportional Attention.
    """

    output_dir = Path(output_dir)

    ptu.set_gpu_mode(True)

    # Build model
    model, variant = load_model(model_path)

    if patch_type == "graph":
        g2tm.graph_segmenter_patch(model, selected_layer, threshold,
                                   prop_attn, iprop_attn)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(ptu.device)

    # Get model config
    patch_size = model.patch_size
    dataset = variant["dataset_kwargs"]["dataset"]
    normalization = variant["dataset_kwargs"]["normalization"]
    crop_size = variant["dataset_kwargs"]["crop_size"]
    n_cls = variant["net_kwargs"]["n_cls"]
    stats = STATS[normalization]

    # Normalize and resize
    img_transform = transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )

    # Open image and process it
    try:
        with open(image_path, "rb") as f:
            img = Image.open(f)
            ori_width, ori_height = img.size
            img_vis = img.resize((crop_size, crop_size))
            img = img.convert("RGB")
        f.close()
        img = img_transform(img)
    except Exception as e:
        print(e)
        raise ValueError(f"Provided image path {image_path}"
                         " is not a valid image file.") from e

    # Open the segmentation groundtruth and process it
    if dataset == "ade20k":
        seg_input_path = (
            image_path.replace("images", "annotations")
                      .replace("jpg", "png")
        )
    elif dataset == "cityscapes":
        seg_input_path = (
            image_path.replace("images", "annotations")
                      .replace("leftImg8bit", "gtFine_labelTrainIds")
        )
    else:
        seg_input_path = "OTHER/DATASET/NOT_SUPPORTED.fail"
    image_name = seg_input_path.split('/')[-1].split('.')[0]

    try:
        with open(seg_input_path, 'rb') as f:
            seg_gt = Image.open(f)
            seg_resized = seg_gt.resize((crop_size, crop_size))
        f.close()
        seg_gt = torch.tensor(np.array(seg_gt))  # pylint: disable=E1101
        seg_resized = np.array(seg_resized)

        # get class ids
        class_list = np.unique(seg_resized)
        class_list = class_list[1:]

        if dataset == "ade20k":
            # Using reduce_zero_label==True => label 0 in segmentation
            # annotations set as 255 (models ignore label 255 in calculating
            # loss) and indices of other labels will minus 1.
            seg_gt = seg_gt - 1
            seg_gt[seg_gt == -1] = 255
    except Exception as e:
        print(e)
        raise ValueError(f"Provided segmentation path {seg_input_path}"
                         " is not a valid segmentation file.") from e

    # Make the image divisible by the patch size
    # NOTE: Process only square image?
    h, w = (
        crop_size - crop_size % patch_size,
        crop_size - crop_size % patch_size,
    )

    # Crop to image size
    img = img[:, :h, :w].unsqueeze(0)
    h_featmap = img.shape[-2] // patch_size
    w_featmap = img.shape[-1] // patch_size

    # Sanity checks
    if not enc and not isinstance(model.decoder, MaskTransformer):
        raise ValueError(
            "Attention maps for decoder are only availabe for MaskTransformer."
            f" Provided model with decoder type: {model.decoder}."
        )

    if not cls:
        if x_patch > w_featmap or y_patch > h_featmap:
            raise ValueError(
                f"Provided patch x: {x_patch} y: {y_patch} is not valid. "
                " Patch should be in the range x:"
                f" [0, {w_featmap}), y: [0, {h_featmap})"
            )
        num_patch = w_featmap * y_patch + x_patch

    if layer_id < 0:
        raise ValueError("Provided layer_id should be positive.")

    if enc and model.encoder.n_layers <= layer_id:
        raise ValueError(
            f"Provided layer_id: {layer_id} is not valid for encoder with"
            f" {model.encoder.n_layers}."
        )

    if not enc and model.decoder.n_layers <= layer_id:
        raise ValueError(
            f"Provided layer_id: {layer_id} is not valid for decoder with"
            f" {model.decoder.n_layers}."
        )

    output_dir = Path(output_dir / image_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get predicted segmentation map
    pred_seg = inference(model, [img.to(ptu.device)],
                         [{'flip': False}], (ori_height, ori_width),
                         variant["inference_kwargs"]["window_size"],
                         variant["inference_kwargs"]["window_stride"], 1)
    pred_seg = pred_seg.argmax(0).cpu()

    # Process input and extract attention maps
    if enc:
        print(f"Generating Attention Mapping for Encoder Layer Id {layer_id}")
        attentions = model.get_attention_map_enc(img.to(ptu.device), layer_id)
        num_extra_tokens = 1 + model.encoder.distilled
        if cls:
            attentions = attentions[0, :, 0, num_extra_tokens:]
        else:
            attentions = attentions[
                0, :, num_patch + num_extra_tokens, num_extra_tokens:  # pylint: disable=E0606
            ]
    else:
        print(f"Generating Attention Mapping for Decoder Layer Id {layer_id}")
        attentions = model.get_attention_map_dec(img.to(ptu.device), layer_id)
        print(attentions.shape)
        if cls:
            attentions = attentions[0, :, -n_cls:, :-n_cls]
        else:
            attentions = attentions[0, :, num_patch, :-n_cls]

    print("Attention map shape: ", attentions.shape)

    # Copying the attention value of the merged tokens to all tokens that have
    # been merged
    n_heads = attentions.shape[0]
    n_ori_tokens = h_featmap * w_featmap
    if attentions.shape[1] != n_ori_tokens:
        idxs = model.encoder.info["source"][0].argmax(dim=0)
        attentions_ = torch.ones(n_heads, n_ori_tokens,  # pylint: disable=E1101
                                 device=attentions.device)
        attentions_[:, :] = attentions[:, idxs]
        print("Attention map shape after unmerging: ", attentions_.shape)
    else:
        attentions_ = attentions

    # Reshape into image shape
    if cls and not enc:
        attentions = attentions_.reshape(n_heads, n_cls, w_featmap, h_featmap)
    else:
        attentions = attentions_.reshape(n_heads, 1, w_featmap, h_featmap)

    # Resize attention maps to match input size
    attentions = (
        F.interpolate(attentions, scale_factor=patch_size, mode="nearest")
         .cpu().numpy()
    )

    # Save Attention map for each head
    class_name = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling',
                  'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk',
                  'person', 'earth', 'door', 'table', 'mountain', 'plant',
                  'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
                  'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
                  'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
                  'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box',
                  'column', 'signboard', 'chest', 'counter', 'sand', 'sink',
                  'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
                  'path', 'stairs', 'runway', 'case', 'pool', 'pillow',
                  'screen', 'stairway', 'river', 'bridge', 'bookcase',
                  'blind', 'coffee', 'toilet', 'flower', 'book', 'hill',
                  'bench', 'countertop', 'stove', 'palm', 'kitchen',
                  'computer', 'swivel', 'boat', 'bar', 'arcade', 'hovel',
                  'bus', 'towel', 'light', 'truck', 'tower', 'chandelier',
                  'awning', 'streetlight', 'booth', 'television', 'airplane',
                  'dirt', 'apparel', 'pole', 'land', 'bannister', 'escalator',
                  'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
                  'ship', 'fountain', 'conveyer', 'canopy', 'washer',
                  'plaything', 'swimming', 'stool', 'barrel', 'basket',
                  'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven',
                  'ball', 'food', 'step', 'tank', 'trade', 'microwave', 'pot',
                  'animal', 'bicycle', 'lake', 'dishwasher', 'screen',
                  'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic',
                  'tray', 'ashcan', 'fan', 'pier', 'crt', 'plate', 'monitor',
                  'bulletin', 'shower', 'radiator', 'glass', 'clock', 'flag']

    for i in range(n_heads):
        base_name = "enc" if enc else "dec"
        head_name = f"{base_name}_layer{layer_id}_attn-head{i}"
        attention_maps_list = attentions[i]
        dir_path = output_dir / f"{base_name}_layer{layer_id}"
        Path.mkdir(dir_path, exist_ok=True)
        if enc:
            attention_map = attention_maps_list[0]
            if cls:
                file_name = head_name + "_cls"
                dir_path /= "cls"
                Path.mkdir(dir_path, exist_ok=True)
            else:
                dir_path /= f"patch_{x_patch}_{y_patch}"
                Path.mkdir(dir_path, exist_ok=True)

            save_attention_map(attention_map, img_vis, dir_path,
                               file_name, cmap)  # pylint: disable=E0606
        else:
            for j in class_list:
                attention_map = attention_maps_list[j-1]
                if cls:
                    file_name = file_name + f"_{class_name[j-1]}"
                    dir_path /= f"cls_{class_name[j-1]}"
                    Path.mkdir(dir_path, exist_ok=True)
                else:
                    dir_path /= f"patch_{x_patch}_{y_patch}"
                    Path.mkdir(dir_path, exist_ok=True)

                save_attention_map(attention_map, img_vis, dir_path,
                                   file_name, cmap)  # pylint: disable=E0606

    # Save input image showing selected patch
    if not cls:
        im_n = torchvision.utils.make_grid(img, normalize=True,
                                           scale_each=True)

        # Compute corresponding X and Y px in the original image
        x_px = x_patch * patch_size
        y_px = y_patch * patch_size
        px_v = einops.repeat(
            torch.tensor([1, 0, 0]),  # pylint: disable=E1101
            "c -> 1 c h w",
            h=patch_size,
            w=patch_size,
        )

        # Draw pixels for selected patch
        im_n[:, y_px:y_px + patch_size, x_px:x_px + patch_size] = px_v

        torchvision.utils.save_image(
            im_n,
            str(output_dir / image_name + ".png"),
        )

    # save the image with merged patches overlayed (2 different visualizations)
    if patch_type == "graph" and layer_id + 1 >= selected_layer:

        source = model.encoder.info["source"]
        n_tokens_left = source.size(1)
        vis_out = g2tm.vis.make_visualization(img_vis, source, patch_size)
        vis_path = (
            output_dir /
            (image_name + f"_vis_{n_tokens_left}_tokens.png")
        )
        vis_out.save(vis_path)
        print(f"{vis_path} saved.")

        vis_out = g2tm.vis.make_overlayed_visualization(img_vis, source,
                                                        patch_size)
        vis_path = (
            output_dir /
            (image_name + f"_vis_{n_tokens_left}_tokens_overlay.png")
        )
        vis_out.save(vis_path)
        print(f"{vis_path} saved.")

    # save the image with the original ViT patch grid
    else:
        vis_out = g2tm.vis.add_grid(img_vis, patch_size)
        vis_path = output_dir / (image_name + "_grid.png")
        vis_out.save(vis_path)
        print(f"{vis_path} saved.")

    # save segmentation maps (groundtruth and predicted)
    vis_seg = g2tm.vis.make_seg_visualization(seg_gt, cmap_file)
    seg_path = output_dir / (image_name + "_seg_gt.png")
    vis_seg.save(seg_path)
    print(f"{seg_path} saved.")

    vis_seg = g2tm.vis.make_seg_visualization(pred_seg, cmap_file)
    seg_path = output_dir / (image_name + "_seg_pred.png")
    vis_seg.save(seg_path)
    print(f"{seg_path} saved.")


if __name__ == "__main__":
    visualize()

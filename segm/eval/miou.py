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


import sys
from pathlib import Path
import shutil

import click
import yaml
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from segm.utils import distributed
from segm.utils.logger import MetricLogger
import segm.utils.torch as ptu
from segm.model.factory import load_model
from segm.data.factory import create_dataset
from segm.metrics import gather_data, compute_metrics
from segm.model.utils import inference
from segm.data.utils import seg_to_rgb, rgb_denormalize, IGNORE_LABEL
from segm import config

import g2tm


def blend_im(im, seg, alpha=0.5):
    """ Blends image.
    """
    pil_im = Image.fromarray(im)
    pil_seg = Image.fromarray(seg)
    im_blend = Image.blend(pil_im, pil_seg, alpha).convert("RGB")
    return np.asarray(im_blend)


def save_im(save_dir, save_name, im, seg_pred, seg_gt, colors, blend,
            normalization):
    """ Saves image.
    """
    seg_rgb = seg_to_rgb(seg_gt[None], colors)
    pred_rgb = seg_to_rgb(seg_pred[None], colors)
    im_unnorm = rgb_denormalize(im, normalization)
    save_dir = Path(save_dir)

    # save images
    im_uint = (im_unnorm.permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)
    seg_rgb_uint = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)
    seg_pred_uint = (255 * pred_rgb.cpu().numpy()).astype(np.uint8)
    for i in range(pred_rgb.shape[0]):
        if blend:
            blend_pred = blend_im(im_uint[i], seg_pred_uint[i])
            blend_gt = blend_im(im_uint[i], seg_rgb_uint[i])
            ims = (im_uint[i], blend_pred, blend_gt)
        else:
            ims = (im_uint[i], seg_pred_uint[i], seg_rgb_uint[i])
        for img, im_dir in zip(
            ims, (save_dir / "input", save_dir / "pred", save_dir / "gt"),
        ):
            pil_out = Image.fromarray(img)
            im_dir.mkdir(exist_ok=True)
            pil_out.save(im_dir / save_name)


def process_batch(
    model, batch, window_size, window_stride, window_batch_size,
):
    """ Runs batch inference.
    """
    ims = batch["im"]
    ims_metas = batch["im_metas"]
    ori_shape = ims_metas[0]["ori_shape"]
    ori_shape = (ori_shape[0].item(), ori_shape[1].item())
    filename = batch["im_metas"][0]["ori_filename"][0]

    model_without_ddp = model
    if ptu.distributed:
        model_without_ddp = model.module
    seg_pred = inference(
        model_without_ddp,
        ims,
        ims_metas,
        ori_shape,
        window_size,
        window_stride,
        window_batch_size,
    )
    seg_pred = seg_pred.argmax(0)
    im = F.interpolate(ims[-1], ori_shape, mode="bilinear")

    return filename, im.cpu(), seg_pred.cpu()


def eval_dataset(
    model,
    multiscale,
    model_dir,
    blend,
    window_size,
    window_stride,
    window_batch_size,
    save_images,
    frac_dataset,
    dataset_kwargs,
):
    """ Evaluate the model on semantic segmentation accuracy metrics.
    """
    db = create_dataset(dataset_kwargs)
    normalization = db.dataset.normalization
    dataset_name = dataset_kwargs["dataset"]
    # im_size = dataset_kwargs["image_size"]
    # cat_names = db.base_dataset.names
    n_cls = db.unwrapped.n_cls
    if multiscale:
        db.dataset.set_multiscale_mode()

    logger = MetricLogger(delimiter="  ")
    header = ""
    print_freq = 50

    ims = {}
    seg_pred_maps = {}
    idx = 0
    for batch in logger.log_every(db, print_freq, header):
        colors = batch["colors"]
        filename, im, seg_pred = process_batch(
            model, batch, window_size, window_stride, window_batch_size,
        )
        ims[filename] = im
        seg_pred_maps[filename] = seg_pred
        idx += 1
        if idx > len(db) * frac_dataset:
            break

    seg_gt_maps = db.dataset.get_gt_seg_maps()
    if save_images:
        save_dir = model_dir / "images"
        if ptu.dist_rank == 0:
            if save_dir.exists():
                shutil.rmtree(save_dir)
            save_dir.mkdir()
        if ptu.distributed:
            torch.distributed.barrier()

        for name in sorted(ims):
            instance_dir = save_dir
            filename = name

            if dataset_name == "cityscapes":
                filename_list = name.split("/")
                instance_dir = instance_dir / filename_list[0]
                filename = filename_list[-1]
                if not instance_dir.exists():
                    instance_dir.mkdir()

            save_im(
                instance_dir,
                filename,
                ims[name],
                seg_pred_maps[name],
                torch.tensor(seg_gt_maps[name]),  # pylint: disable=E1101
                colors,
                blend,
                normalization,
            )
        if ptu.dist_rank == 0:
            shutil.make_archive(save_dir, "zip", save_dir)
            # shutil.rmtree(save_dir)
            print(f"Saved eval images in {save_dir}.zip")

    if ptu.distributed:
        torch.distributed.barrier()
        seg_pred_maps = gather_data(seg_pred_maps)

    scores = compute_metrics(
        seg_pred_maps,
        seg_gt_maps,
        n_cls,
        ignore_index=IGNORE_LABEL,
        ret_cat_iou=True,
        distributed=ptu.distributed,
    )

    if ptu.dist_rank == 0:
        scores["inference"] = ("single_scale" if not multiscale
                               else "multi_scale")
        suffix = "ss" if not multiscale else "ms"
        scores["cat_iou"] = np.round(100 * scores["cat_iou"], 2).tolist()
        for k, v in scores.items():
            if k != "cat_iou":
                if k != "inference":
                    scores[k] = v.item()
                print(f"{k}: {scores[k]}")
        scores_str = yaml.dump(scores)
        with open(model_dir / f"scores_{suffix}.yml", "w", encoding='utf-8') as f:
            f.write(scores_str)


@click.command()
@click.argument("model_path", type=str)
@click.argument("dataset_name", type=str)
@click.option("--im-size", default=None, type=int)
@click.option("--multiscale/--singlescale", default=False, is_flag=True)
@click.option("--blend/--no-blend", default=True, is_flag=True)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--window-batch-size", default=4, type=int)
@click.option("--save-images/--no-save-images", default=False, is_flag=True)
@click.option("--frac-dataset", default=1.0, type=float)
@click.option("--patch-type", default="pure", type=str)
@click.option("--selected-layer", default=1, type=int)
@click.option("--threshold", default=0.88, type=float)
@click.option("--prop-attn/--no-prop-attn", default=False, is_flag=True)
@click.option("--iprop-attn/--no-iprop-attn", default=False, is_flag=True)
def main(model_path, dataset_name, im_size, multiscale, blend,
         window_size, window_stride, window_batch_size, save_images,
         frac_dataset, patch_type, selected_layer, threshold, prop_attn,
         iprop_attn):
    """ Compute semantic segmentation accuracy metrics of the model.
    """
    model_dir = Path(model_path).parent

    # start distributed mode
    ptu.set_gpu_mode(True)
    distributed.init_process()

    model, variant = load_model(model_path)

    if patch_type == "graph":
        g2tm.graph_segmenter_patch(model, selected_layer, threshold,
                                   prop_attn, iprop_attn)

    patch_size = model.patch_size
    model.eval()
    model.to(ptu.device)
    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device],
                    find_unused_parameters=True)

    cfg = config.load_config()
    dataset_cfg = cfg["dataset"][dataset_name]
    normalization = variant["dataset_kwargs"]["normalization"]
    if im_size is None:
        im_size = dataset_cfg.get("im_size",
                                  variant["dataset_kwargs"]["image_size"])
    if window_size is None:
        window_size = variant["dataset_kwargs"]["crop_size"]
    if window_stride is None:
        window_stride = variant["dataset_kwargs"]["crop_size"] - 32

    dataset_kwargs = {
        "dataset": dataset_name,
        "image_size": im_size,
        "crop_size": im_size,
        "patch_size": patch_size,
        "batch_size": 1,
        "num_workers": 10,
        "split": 'val',
        "normalization": normalization,
        "crop": False,
        "rep_aug": False,
    }

    eval_dataset(
        model,
        multiscale,
        model_dir,
        blend,
        window_size,
        window_stride,
        window_batch_size,
        save_images,
        frac_dataset,
        dataset_kwargs,
    )

    distributed.barrier()
    distributed.destroy_process()
    sys.exit(1)


if __name__ == "__main__":
    main()

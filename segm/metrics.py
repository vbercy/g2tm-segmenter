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


import os
import pickle as pkl
from pathlib import Path
import tempfile
import shutil

import numpy as np
from mmseg.core import mean_iou

import torch
import torch.distributed as dist

import segm.utils.torch as ptu


def accuracy(output, target, topk=(1,)):
    """Computes the ImageNet classifcation accuracy over the k top predictions
    for the specified values of k borrowed from:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_k /= batch_size
            res.append(correct_k)
        return res


# Segmentation mean IoU
# based on collect_results_cpu
# https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/apis/test.py#L160-L200


def gather_data(seg_pred, tmp_dir=None):
    """Distributed data gathering prediction and ground truth are stored in a
    common tmp directory and loaded on the master node to compute metrics.
    """
    if tmp_dir is None:
        tmpprefix = os.path.expandvars("./temp")
    else:
        tmpprefix = os.path.expandvars(tmp_dir)
    max_len = 512
    # 32 is whitespace
    dir_tensor = torch.full((max_len,), 32, dtype=torch.uint8,  # pylint: disable=E1101
                            device=ptu.device)
    if ptu.dist_rank == 0:
        tmpdir = tempfile.mkdtemp(prefix=tmpprefix)
        tmpdir = torch.tensor(  # pylint: disable=E1101
            bytearray(tmpdir.encode()), dtype=torch.uint8, device=ptu.device
        )
        dir_tensor[: len(tmpdir)] = tmpdir
    # broadcast tmpdir from 0 to to the other nodes
    dist.broadcast(dir_tensor, 0)
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    tmpdir = Path(tmpdir)
    # Save results in temp file and load them on main process
    tmp_file = tmpdir / f"part_{ptu.dist_rank}.pkl"
    with open(tmp_file, "wb") as f:
        pkl.dump(seg_pred, f)
    dist.barrier()
    seg_pred = {}
    if ptu.dist_rank == 0:
        for i in range(ptu.world_size):
            with open(tmpdir / f"part_{i}.pkl", "rb") as f:
                part_seg_pred = pkl.load(f)
            seg_pred.update(part_seg_pred)
        shutil.rmtree(tmpdir)
    return seg_pred


def compute_metrics(
    seg_pred,
    seg_gt,
    n_cls,
    ignore_index=None,
    ret_cat_iou=False,
    distributed=False,
):
    """ Compute accuracy metrics for semantic segmentation task.
    """
    ret_metrics_mean = torch.zeros(3, dtype=float, device=ptu.device)  # pylint: disable=E1101
    if ptu.dist_rank == 0:
        list_seg_pred = []
        list_seg_gt = []
        keys = sorted(seg_pred.keys())
        for k in keys:
            list_seg_pred.append(np.asarray(seg_pred[k]))
            list_seg_gt.append(np.asarray(seg_gt[k]))
        ret_metrics = mean_iou(
            results=list_seg_pred,
            gt_seg_maps=list_seg_gt,
            num_classes=n_cls,
            ignore_index=ignore_index,
        )
        ret_metrics = [ret_metrics["aAcc"], ret_metrics["Acc"],
                       ret_metrics["IoU"]]
        ret_metrics_mean = torch.tensor(  # pylint: disable=E1101
            [
                np.round(np.nanmean(ret_metric.astype(np.float64)) * 100, 2)
                for ret_metric in ret_metrics
            ],
            dtype=float,
            device=ptu.device,
        )
        cat_iou = ret_metrics[2]
    # broadcast metrics from 0 to all nodes
    if distributed:
        dist.broadcast(ret_metrics_mean, 0)
    pix_acc, mean_acc, miou = ret_metrics_mean
    ret = {"pixel_accuracy": pix_acc, "mean_accuracy": mean_acc, "mean_iou": miou}
    if ret_cat_iou and ptu.dist_rank == 0:
        ret["cat_iou"] = cat_iou  # pylint: disable=E0606
    return ret

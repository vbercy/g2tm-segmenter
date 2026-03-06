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


import yaml

import torch
import torchvision.transforms.functional as F

IGNORE_LABEL = 255
STATS = {
    "vit": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "deit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}


def seg_to_rgb(seg, colors):
    """ From segmentation labels to RGB colors.
    """
    im = torch.zeros((seg.shape[0], seg.shape[1], seg.shape[2], 3)).float()  # pylint: disable=E1101
    cls = torch.unique(seg)
    for cl in cls:
        color = colors[int(cl)]
        if len(color.shape) > 1:
            color = color[0]
        im[seg == cl] = color
    return im


def dataset_cat_description(path, cmap=None):
    """ Get names and colors associated to each label.
    """
    with open(path, "r", encoding='utf-8') as f:
        desc = yaml.load(f, Loader=yaml.FullLoader)
    colors = {}
    names = []
    for cat in desc:
        names.append(cat["name"])
        if "color" in cat:
            colors[cat["id"]] = torch.tensor(cat["color"]).float() / 255  # pylint: disable=E1101
        else:
            colors[cat["id"]] = torch.tensor(cmap[cat["id"]]).float()  # pylint: disable=E1101
    colors[IGNORE_LABEL] = torch.tensor([0.0, 0.0, 0.0]).float()  # pylint: disable=E1101
    return names, colors


def rgb_normalize(x, stats):
    """
    x : C x *
    x in [0, 1]
    """
    return F.normalize(x, stats["mean"], stats["std"])


def rgb_denormalize(x, stats):
    """
    x : N x C x *
    x in [-1, 1]
    """
    mean = torch.tensor(stats["mean"])  # pylint: disable=E1101
    std = torch.tensor(stats["std"])  # pylint: disable=E1101
    for i in range(3):
        x[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return x

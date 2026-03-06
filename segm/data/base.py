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

import numpy as np
import mmcv
from mmcv.utils import Config
from mmseg.datasets import build_dataset

from torch.utils.data import Dataset

from segm.data.utils import STATS, IGNORE_LABEL


class BaseMMSeg(Dataset):
    """ Base dataset class.
    """
    def __init__(
        self,
        image_size,
        crop_size,
        split,
        config_path,
        normalization,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.crop_size = crop_size
        self.split = split
        self.normalization = STATS[normalization].copy()
        self.ignore_label = None
        for k, v in self.normalization.items():
            v = np.round(255 * np.array(v), 2)
            self.normalization[k] = tuple(v)
        print(f"Use normalization: {self.normalization}")

        config = Config.fromfile(config_path)

        self.ratio = config.max_ratio
        self.dataset = None
        self.config = self.update_default_config(config)
        self.dataset = build_dataset(getattr(self.config.data,
                                             f"{self.split}"))

    def update_default_config(self, config):
        """ Update default configuration.
        """
        train_splits = ["train", "trainval"]
        if self.split in train_splits:
            config_pipeline = getattr(config, "train_pipeline")
        else:
            config_pipeline = getattr(config, f"{self.split}_pipeline")

        img_scale = (self.ratio * self.image_size, self.image_size)
        if self.split not in train_splits:
            assert config_pipeline[1]["type"] == "MultiScaleFlipAug"
            config_pipeline = config_pipeline[1]["transforms"]
        for i, op in enumerate(config_pipeline):
            op_type = op["type"]
            if op_type == "Resize":
                op["img_scale"] = img_scale
            elif op_type == "RandomCrop":
                op["crop_size"] = (
                    self.crop_size,
                    self.crop_size,
                )
            elif op_type == "Normalize":
                op["mean"] = self.normalization["mean"]
                op["std"] = self.normalization["std"]
            elif op_type == "Pad":
                op["size"] = (self.crop_size, self.crop_size)
            config_pipeline[i] = op
        if self.split == "train":
            config.data.train.pipeline = config_pipeline
        elif self.split == "trainval":
            config.data.trainval.pipeline = config_pipeline
        elif self.split == "val":
            config.data.val.pipeline[1]["img_scale"] = img_scale
            config.data.val.pipeline[1]["transforms"] = config_pipeline
        elif self.split == "fps_val":
            config.data.val.pipeline[1]["img_scale"] = img_scale
            config.data.val.pipeline[1]["transforms"] = config_pipeline
            config.data.val.pipeline[1]["flip"] = False
            config.data.fps_val.test_mode = True
        elif self.split == "test":
            config.data.test.pipeline[1]["img_scale"] = img_scale
            config.data.test.pipeline[1]["transforms"] = config_pipeline
            config.data.test.test_mode = True
        else:
            raise ValueError(f"Unknown split: {self.split}")
        return config

    def set_multiscale_mode(self):
        """ Set multi-scale image ratios.
        """
        self.config.data.val.pipeline[1]["img_ratios"] = [
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
        ]
        self.config.data.val.pipeline[1]["flip"] = True
        self.config.data.test.pipeline[1]["img_ratios"] = [
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
        ]
        self.config.data.test.pipeline[1]["flip"] = True
        self.dataset = build_dataset(getattr(self.config.data,
                                             f"{self.split}"))

    def __getitem__(self, idx):
        data = self.dataset[idx]

        train_splits = ["train", "trainval"]

        if self.split in train_splits:
            im = data["img"].data
            seg = data["gt_semantic_seg"].data.squeeze(0)
        else:
            im = [im.data for im in data["img"]]
            seg = None

        out = {"im": im}
        if self.split in train_splits:
            out["segmentation"] = seg
        else:
            im_metas = [meta.data for meta in data["img_metas"]]
            out["im_metas"] = im_metas
            out["colors"] = self.colors  # pylint: disable=E1101

        return out

    def get_gt_seg_maps(self):
        """ Get groundtruth segmentation maps.
        """
        dataset = self.dataset
        gt_seg_maps = {}
        for img_info in dataset.img_infos:
            seg_map = Path(dataset.ann_dir) / img_info["ann"]["seg_map"]
            gt_seg_map = mmcv.imread(seg_map, flag="unchanged",
                                     backend="pillow")
            gt_seg_map[gt_seg_map == self.ignore_label] = IGNORE_LABEL
            if self.reduce_zero_label:  # pylint: disable=E1101
                gt_seg_map[gt_seg_map != IGNORE_LABEL] -= 1
            gt_seg_maps[img_info["filename"]] = gt_seg_map
        return gt_seg_maps

    def __len__(self):
        return len(self.dataset)

    @property
    def unwrapped(self):
        """ Unwrap.
        """
        return self

    def set_epoch(self, epoch):
        """ Set number of epochs.
        """

    def get_diagnostics(self, logger):
        """ Get diagnostics from logger.
        """

    def get_snapshot(self):
        """ Get snapshot.
        """
        return {}

    def end_epoch(self, epoch):
        """ Get end epoch.
        """

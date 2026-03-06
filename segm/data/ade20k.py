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

from segm.data.base import BaseMMSeg
from segm.data import utils
from segm.config import dataset_dir


ADE20K_CONFIG_PATH = Path(__file__).parent / "config" / "ade20k.py"
ADE20K_CATS_PATH = Path(__file__).parent / "config" / "ade20k.yml"


class ADE20KSegmentation(BaseMMSeg):
    """ ADE20K dataset class.
    """
    def __init__(self, image_size, crop_size, split, **kwargs):
        super().__init__(
            image_size,
            crop_size,
            split,
            ADE20K_CONFIG_PATH,
            **kwargs,
        )
        self.names, self.colors = utils.dataset_cat_description(
            ADE20K_CATS_PATH
        )
        self.n_cls = 150
        self.ignore_label = 0
        self.reduce_zero_label = True

    def update_default_config(self, config):
        root_dir = dataset_dir()
        path = Path(root_dir)  # / "ade20k"
        config.data_root = path
        if self.split == "train":
            config.data.train.data_root = path  # / "ADEChallengeData2016"
        elif self.split == "trainval":
            config.data.trainval.data_root = path  # / "ADEChallengeData2016"
        elif self.split == "val":
            config.data.val.data_root = path  # / "ADEChallengeData2016"
        elif self.split == "fps_val":
            config.data.fps_val.data_root = path  # / "ADEChallengeData2016"
        elif self.split == "test":
            config.data.test.data_root = path  # / "release_test"
        config = super().update_default_config(config)
        return config

    def test_post_process(self, labels):
        """ Test post-processing.
        """
        return labels + 1

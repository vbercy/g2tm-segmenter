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

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

from segm.data import utils


class ImagenetDataset(Dataset):
    """ ImageNet dataset class.
    """
    def __init__(
        self,
        root_dir,
        image_size=224,
        crop_size=224,
        split="train",
        normalization="vit",
    ):
        super().__init__()
        assert image_size[0] == image_size[1]

        self.path = Path(root_dir) / split
        self.crop_size = crop_size
        self.image_size = image_size
        self.split = split
        self.normalization = normalization

        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.crop_size,
                                                 interpolation=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size[0] + 32, interpolation=3),
                    transforms.CenterCrop(self.crop_size),
                    transforms.ToTensor(),
                ]
            )

        self.base_dataset = datasets.ImageFolder(self.path, self.transform)
        self.n_cls = 1000

    @property
    def unwrapped(self):
        """ Unwrap.
        """
        return self

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        im, target = self.base_dataset[idx]
        im = utils.rgb_normalize(im, self.normalization)
        return {"im": im, "target": target}

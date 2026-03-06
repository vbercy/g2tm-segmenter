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


import segm.utils.torch as ptu
from segm.data import ImagenetDataset
from segm.data import ADE20KSegmentation
from segm.data import PascalContextDataset
from segm.data import CityscapesDataset
from segm.data import Loader


def create_dataset(dataset_kwargs):
    """ Create a dataset from configuration dictionary.
    """
    dataset_kwargs = dataset_kwargs.copy()
    dataset_name = dataset_kwargs.pop("dataset")
    batch_size = dataset_kwargs.pop("batch_size")
    num_workers = dataset_kwargs.pop("num_workers")
    split = dataset_kwargs.pop("split")

    # load dataset_name
    if dataset_name == "imagenet":
        dataset_kwargs.pop("patch_size")
        dataset = ImagenetDataset(split=split, **dataset_kwargs)
    elif dataset_name == "ade20k":
        dataset = ADE20KSegmentation(split=split, **dataset_kwargs)
    elif dataset_name == "pascal_context":
        dataset = PascalContextDataset(split=split, **dataset_kwargs)
    elif dataset_name == "cityscapes":
        dataset = CityscapesDataset(split=split, **dataset_kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} is unknown.")

    dataset = Loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=ptu.distributed,
        split=split,
    )
    return dataset

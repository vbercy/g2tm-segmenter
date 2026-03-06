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
import shutil
import tarfile

import click
from tqdm import tqdm

import torch

from segm.utils.download import download


def download_pcontext(path, overwrite=False):
    """Download PASCAL Context dataset.
    """
    _aug_download_urls = [
        (
            "https://www.dropbox.com/s/wtdibo9lb2fur70/VOCtrainval_03-May-2010.tar?dl=1",
            "VOCtrainval_03-May-2010.tar",
            "bf9985e9f2b064752bf6bd654d89f017c76c395a",
        ),
        (
            "https://codalabuser.blob.core.windows.net/public/trainval_merged.json",
            "",
            "169325d9f7e9047537fedca7b04de4dddf10b881",
        ),
        (
            "https://hangzh.s3.amazonaws.com/encoding/data/pcontext/train.pth",
            "",
            "4bfb49e8c1cefe352df876c9b5434e655c9c1d07",
        ),
        (
            "https://hangzh.s3.amazonaws.com/encoding/data/pcontext/val.pth",
            "",
            "ebedc94247ec616c57b9a2df15091784826a7b0c",
        ),
    ]
    download_dir = path / "downloads"

    download_dir.mkdir(parents=True, exist_ok=True)

    for url, filename, checksum in _aug_download_urls:
        filename = download(
            url,
            path=str(download_dir / filename),
            overwrite=overwrite,
            sha1_hash=checksum,
        )
        # extract
        if Path(filename).suffix == ".tar":
            with tarfile.open(filename) as tar:
                tar.extractall(path=str(path))
        else:
            shutil.move(
                filename,
                str(path / "VOCdevkit" / "VOC2010" / Path(filename).name),
            )


@click.command(help="Initialize PASCAL Context dataset.")
@click.argument("download_dir", type=str)
def main(download_dir):
    """Prepare PASCAL Context dataset.
    """
    dataset_dir = Path(download_dir) / "pcontext"

    download_pcontext(dataset_dir, overwrite=False)

    devkit_path = dataset_dir / "VOCdevkit"
    out_dir = devkit_path / "VOC2010" / "SegmentationClassContext"
    imageset_dir = (devkit_path / "VOC2010" /
                    "ImageSets" / "SegmentationContext")

    out_dir.mkdir(parents=True, exist_ok=True)
    imageset_dir.mkdir(parents=True, exist_ok=True)

    train_torch_path = devkit_path / "VOC2010" / "train.pth"
    val_torch_path = devkit_path / "VOC2010" / "val.pth"

    train_dict = torch.load(str(train_torch_path))

    train_list = []
    for idx, label in tqdm(train_dict.items()):
        idx = str(idx)
        new_idx = idx[:4] + "_" + idx[4:]
        train_list.append(new_idx)
        label_path = out_dir / f"{new_idx}.png"
        label.save(str(label_path))

    with open(str(imageset_dir / "train.txt"), "w", encoding='utf-8') as f:
        f.writelines(line + "\n" for line in sorted(train_list))

    val_dict = torch.load(str(val_torch_path))

    val_list = []
    for idx, label in tqdm(val_dict.items()):
        idx = str(idx)
        new_idx = idx[:4] + "_" + idx[4:]
        val_list.append(new_idx)
        label_path = out_dir / f"{new_idx}.png"
        label.save(str(label_path))

    with open(str(imageset_dir / "val.txt"), "w", encoding='utf-8') as f:
        f.writelines(line + "\n" for line in sorted(val_list))


if __name__ == "__main__":
    main()

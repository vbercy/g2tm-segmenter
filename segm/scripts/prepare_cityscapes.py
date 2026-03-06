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
import zipfile
from pathlib import Path

import click
import mmcv
from cityscapesscripts.preparation.json2labelImg import json2labelImg

USERNAME = None
PASSWORD = None


def download_cityscapes(path, username, password):
    """ Download Cityscapes dataset.
    """
    _city_download_urls = [
        ("gtFine_trainvaltest.zip",
            "99f532cb1af174f5fcc4c5bc8feea8c66246ddbc"),
        ("leftImg8bit_trainvaltest.zip",
            "2c0b77ce9933cc635adda307fbba5566f5d9d404"),
    ]
    download_dir = path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    os.system(
        "wget --keep-session-cookies --save-cookies=cookies.txt --post-data"
        f" 'username={username}&password={password}&submit=Login'"
        f" https://www.cityscapes-dataset.com/login/ -P {download_dir}"
    )

    if not (download_dir / "gtFine_trainvaltest.zip").is_file():
        os.system(
            "wget --load-cookies cookies.txt --content-disposition"
            " https://www.cityscapes-dataset.com/file-handling/?packageID=1"
            f" -P {download_dir}"
        )

    if not (download_dir / "leftImg8bit_trainvaltest.zip").is_file():
        os.system(
            "wget --load-cookies cookies.txt --content-disposition"
            " https://www.cityscapes-dataset.com/file-handling/?packageID=3"
            f" -P {download_dir}"
        )

    for filename, _ in _city_download_urls:
        # extract
        with zipfile.ZipFile(str(download_dir / filename), "r") as zip_ref:
            zip_ref.extractall(path=path)
        print("Extracted", filename)


def install_cityscapes_api():
    """ Installing cityscapesscripts librairy.
    """
    os.system("pip install cityscapesscripts")
    try:
        import cityscapesscripts  # noqa: F401
    except ImportError:
        print(
            "Installing Cityscapes API failed, please install it manually."
        )


def convert_json_to_label(json_file):
    """ Convert json file to Cityscapes label.
    """
    label_file = json_file.replace("_polygons.json", "_labelTrainIds.png")
    json2labelImg(json_file, label_file, "trainIds")


@click.command(help="Initialize Cityscapes dataset.")
@click.argument("download_dir", type=str)
@click.option("--username", default=USERNAME, type=str)
@click.option("--password", default=PASSWORD, type=str)
@click.option("--nproc", default=10, type=int)
def main(
    download_dir,
    username,
    password,
    nproc,
):
    """ Prepare Cityscapes dataset.
    """

    dataset_dir = Path(download_dir) / "cityscapes"

    if username is None or password is None:
        raise ValueError(
            "You must indicate your username and password either in the script"
            " variables or by passing options --username and --pasword."
        )

    download_cityscapes(dataset_dir, username, password)

    install_cityscapes_api()

    gt_dir = dataset_dir / "gtFine"

    poly_files = []
    for poly in mmcv.scandir(str(gt_dir), "_polygons.json", recursive=True):
        poly_file = str(gt_dir / poly)
        poly_files.append(poly_file)
    mmcv.track_parallel_progress(convert_json_to_label, poly_files, nproc)

    split_names = ["train", "val", "test"]

    for split in split_names:
        filenames = []
        for poly in mmcv.scandir(str(gt_dir / split), "_polygons.json",
                                 recursive=True):
            filenames.append(poly.replace("_gtFine_polygons.json", ""))
        name = f"{split}.txt"
        with open(str(dataset_dir / name), "w", encoding='utf-8') as f:
            f.writelines(f + "\n" for f in filenames)


if __name__ == "__main__":
    main()

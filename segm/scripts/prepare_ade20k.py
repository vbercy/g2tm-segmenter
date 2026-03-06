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


import zipfile
from pathlib import Path

import click

from segm.utils.download import download


def download_ade(path, overwrite=False):
    """Download ADE20K dataset.
    """
    _aug_download_urls = [
        (
            "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",  # noqa: E501
            "219e1696abb36c8ba3a3afe7fb2f4b4606a897c7",
        ),
        (
            "http://data.csail.mit.edu/places/ADEchallenge/release_test.zip",
            "e05747892219d10e9243933371a497e905a4860c",
        ),
    ]
    download_dir = path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    for url, checksum in _aug_download_urls:
        filename = download(
            url, path=str(download_dir),
            overwrite=overwrite, sha1_hash=checksum
        )
        # extract
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(path=str(path))


@click.command(help="Initialize ADE20K dataset.")
@click.argument("download_dir", type=str)
def main(download_dir):
    """ Prepare ADE20K dataset.
    """
    dataset_dir = Path(download_dir) / "ade20k"
    download_ade(dataset_dir, overwrite=False)


if __name__ == "__main__":
    main()

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
from pathlib import Path

import yaml


def load_config():
    """ Load YAML configuration file containing model parameters.
    """
    return yaml.load(
        open(Path(__file__).parent / "config.yml", "r", encoding='utf-8'),
        Loader=yaml.FullLoader
    )


def check_os_environ(key, use):
    """ Check if variable is defined as an environment variable.
    """
    if key not in os.environ:
        raise ValueError(
            f"{key} is not defined in the os variables, it is required"
            f"for {use}."
        )


def dataset_dir():
    """ Check if DATASET is defined as an environment variable.
    """
    check_os_environ("DATASET", "data loading")
    return os.environ["DATASET"]

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


__version__ = "0.0.1c"


def parse_version_info(version_str):
    """Parse a version string into a tuple.

    Args:
        version_str (str): The version string.
    Returns:
        tuple[int | str]: The version info as tuple, e.g.: "1.3.0" is parsed
            into (1, 3, 0) and "2.0.0rc1" is parsed into (2, 0, 0, 'rc1')
    """
    version_infos = []
    for x in version_str.split("."):
        if x.isdigit():
            version_infos.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_infos.append(int(patch_version[0]))
            version_infos.append(f"rc{patch_version[1]}")
    return tuple(version_infos)


version_info = parse_version_info(__version__)

__all__ = ["__version__", "parse_version_info", "version_info"]

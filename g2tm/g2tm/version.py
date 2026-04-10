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


__version__ = "0.1b"


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

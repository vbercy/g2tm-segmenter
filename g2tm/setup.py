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


from setuptools import setup


def get_version():
    """ Returns the segm package version.
    """
    version = None
    with open('./g2tm/version.py', encoding='utf-8') as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break
    return version

def readme():
    """ Returns README file as string.
    """
    with open('../README.md', encoding='utf-8') as f:
        content = f.read()
    return content


setup(
    name="g2tm",
    version=get_version(),
    author="Victor BERCY",
    description="G2TM: Single Module Graph-Guided Token Merging for Efficient"
                "for Semantic Segmentation",
    install_requires=[
        "numpy<2",
        "torch<2",
        "torchvision",
        "opencv-python",
        "pillow",
        "scipy",
        "networkx",
        "nx-cugraph-cu11",
    ],
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=["g2tm"],
)

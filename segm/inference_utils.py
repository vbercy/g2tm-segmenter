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

# Modifications based on code from Narges Norouzi et al. (ALGM)


import os
import warnings

from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


warnings.filterwarnings("ignore")


def get_dataset_inference_path(dataset_name: str, root_dir: str):
    """Get validation data path from the dataset name.

    Args:
        dataset_name (str): Name of the dataset.
        root_dir (str): Root directory of the dataset.

    Returns:
        dataset_path (str): Path to the validation data.
        dataset_path_txt (str): Path to specific text file for Pascal Context.
    """
    dataset_path, dataset_path_txt = None, None
    if dataset_name == 'ade20k':
        dataset_path = root_dir + '/images/validation/'
    elif dataset_name == 'cityscapes':
        dataset_path = root_dir + '/leftImg8bit/val/'
    elif dataset_name == 'pascal_context':
        dataset_path_txt = root_dir + '/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
        dataset_path = root_dir + '/VOCdevkit/VOC2012/JPEGImages/'

    return dataset_path, dataset_path_txt


class InferenceDataset(Dataset):
    """PyTorch Dataset for inference.

    Args:
        root_dir (str): Root directory of the dataset.
        transform (transforms.Compose): Transformations to apply to the images.
        txt_file (str): Path to optional text file (for Pascal Context only).
    """
    def __init__(self, root_dir: str, transform: transforms.Compose = None,
                 txt_file: str = None):
        self.root_dir = root_dir
        self.transform = transform

        # If txt_file is provided, read image names from it
        if txt_file:
            with open(txt_file, 'r', encoding='utf-8') as file:
                self.image_files = [
                    os.path.join(root_dir, line.strip() + '.jpg')
                    for line in file.readlines()
                ]
        else:
            # Otherwise, load all image paths from root_dir and subfolders
            self.image_files = self._load_image_paths(root_dir)

    def _load_image_paths(self, root_dir):
        image_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    image_files.append(os.path.join(dirpath, file))
        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def dataset_prepare(dataset_path: str, dataset_txt_path: str,
                    stats: dict, batch_size: int, input_size: int,
                    shuffle: bool = True) -> Dataset:
    """Inference dataset.

    This function creates a dataset for inference from the dataset*
    directory path.

    Args:
        dataset_path (str): Path to the validation data.
        dataset_path_txt (str): Path to specific text file for Pascal Context.
        stats (dict): Normalization statistics for the dataset.
        batch_size (int): Number of image per batch.
        input_size (int): Size (height and width) of the input image in the
            dataloader.
        shuffle (bool): Whether to shuffle the order of the images.

    Returns:
        validation_loader (Dataset): PyTorch dataloader.
    """
    validation_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"])
    ])

    if dataset_txt_path is None:
        validation_dataset = InferenceDataset(root_dir=dataset_path,
                                              transform=validation_transforms)
    else:
        validation_dataset = InferenceDataset(root_dir=dataset_path,
                                              transform=validation_transforms,
                                              txt_file=dataset_txt_path)

    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=shuffle)
    return validation_loader

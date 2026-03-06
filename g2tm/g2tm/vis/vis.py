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

# Modifications based on code from Daniel Bolya et al. (ToMe)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------


import random
from typing import List, Tuple

import yaml
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

import torch
import torch.nn.functional as F


def generate_colormap(n: int,
                      seed: int = 0) -> List[Tuple[float, float, float]]:
    """Generates a colormap with N floating-point elements.

    Args:
        n (int): Number of colors to generate.
        seed (int): Random seed.

    Returns:
        list[tuple[float]]: List of RGB colors.
    """
    random.seed(seed)

    def generate_color():
        return (random.random(),
                random.random(),
                random.random())

    return [generate_color() for _ in range(n)]


def generate_colormap_int(n: int, seed: int = 0) -> List[Tuple[int, int, int]]:
    """Generates a colormap with N integer elements.

    Args:
        n (int): Number of colors to generate.
        seed (int): Random seed.

    Returns:
        list[tuple[int]]: List of RGB colors.
    """
    random.seed(seed)

    def generate_color():
        return (random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255))

    return [generate_color() for _ in range(n)]


def make_visualization(img: Image.Image, source: torch.Tensor,
                       patch_size: int = 16) -> Image.Image:
    """Overlay of the fused patches on the original image.

    This function creates a visualization of the fused patches on top of the
    original image, like in the paper.

    Args:
        img (PIL.Image.Image): Resized image from the dataset.
        source (torch.Tensor): Matrix indicating which tokens have been fused
            with another.
        patch_size (int): Height and width of an image patch in pixels.

    Returns:
        PIL.Image.Image: The expected visualization with the same size as the
            input.
    """
    img = np.array(img.convert("RGB")) / 255.0
    source = source.detach().cpu()

    h, w, _ = img.shape
    ph = h // patch_size
    pw = w // patch_size

    print(f"Number of tokens left: {source.size(1)}.")

    vis = source.argmax(dim=1)
    num_groups = vis.max().item() + 1

    cmap = generate_colormap(num_groups)
    vis_img = 0

    for i in range(1, num_groups):
        mask = (vis == i).float().view(1, 1, ph, pw)
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)  # pylint: disable=E1121

    # Convert back into a PIL image
    vis_img = Image.fromarray(np.uint8(vis_img * 255))

    return vis_img


def make_overlayed_visualization(img: Image.Image, source: torch.Tensor,
                                 patch_size: int = 16) -> Image.Image:
    """Overlay of the fused patches on the original image.

    This function creates a variant of the visualization of the fused patches.
    A grid overlaying the original image detours the fused patches.

    Args:
        img (PIL.Image.Image): Resized image from the dataset.
        source (torch.Tensor): Matrix indicating which tokens have been fused
            with another.
        patch_size (int): Height and width of an image patch in pixels.

    Returns:
        PIL.Image.Image: The expected visualization with the same size as the
            input.
    """
    w, h = img.size
    n_w, n_h = w // patch_size, h // patch_size

    vis = source[0].argmax(dim=0)
    num_groups = vis.max().item() + 1

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cmap = generate_colormap_int(num_groups)

    for i in range(1, num_groups):
        color_rgb = cmap[i]
        color = (*color_rgb, 0)
        border_color = (0, 0, 255, 180)

        mask = np.zeros((n_h, n_w), dtype=np.uint8)
        group = (vis == i).nonzero()

        for idx in group:
            row, col = divmod(idx.item(), n_w)
            left = col * patch_size
            top = row * patch_size
            draw.rectangle(
                [left, top, left + patch_size, top + patch_size], fill=color
            )
            mask[row, col] = 255

        mask_img = Image.fromarray(mask).resize((w, h), resample=Image.NEAREST)
        mask_np = np.array(mask_img)
        contours, _ = cv2.findContours(  # pylint: disable=E1101
            mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE  # pylint: disable=E1101
        )

        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and len(contour) >= 2:
                draw.line(
                    contour.tolist() + [contour.tolist()[0]],
                    fill=border_color, width=1
                )

    blended = Image.alpha_composite(img.convert("RGBA"), overlay)

    return blended


def make_seg_visualization(seg: torch.Tensor, cmap_file: str) -> Image:
    """Generate an image representing the segmentation map.

    This function returns a segmentation map that contains a RGB color
    corresponding to its class for each pixel of the image, instead of the
    class ID.

    Args:
        seg (torch.Tensor): Matrix of class IDs for each pixels.
        cmap_file (str): Path to the YAML file storing the correspondances
            between class ID and colors.

    Returns:
        vis_seg (torch.Tensor): The segmentation map (H, W, 3)).
    """
    h, w = seg.shape[-2:]
    vis_seg = torch.zeros((h, w, 3), dtype=int)  # pylint: disable=E1101

    classes = torch.unique(seg).tolist()
    with open(cmap_file, 'r', encoding='utf-8') as f:
        data_config = yaml.full_load(f)
    f.close()
    cmap = dict(
        zip(
            list(map(lambda x: x['id'], data_config)),
            list(map(lambda x: x['color'], data_config))
        )
    )
    cmap[255] = [0, 0, 0]

    for c in classes:
        color = torch.tensor(cmap[c])  # pylint: disable=E1101
        vis_seg[seg == c] = color

    # Convert back into a PIL image
    vis_seg = Image.fromarray(np.uint8(vis_seg))

    return vis_seg


def add_grid(image: Image.Image, grid_size: int = 16,
             grid_color: tuple = (128, 128, 128),
             thickness: int = 1) -> Image.Image:
    """ Overlay of the original ViT patches on the image.

    This function adds a regular grid of the specified size to the input image.

    Args:
        image (PIL.Image.Image): Input image.
        grid_size (int): Size of each grid cell (i.e.: size of an image patch
            in pixels).
        grid_color (tuple[int]): Color of the grid lines (default is gray).
        thickness (int): Thickness of the grid lines.

    Returns:
        image_with_grid (PIL.Image.Image): Regular grid overlaying the input
            image.
    """
    # Create a copy of the input image to avoid modifying the original image
    image_with_grid = image.copy()
    draw = ImageDraw.Draw(image_with_grid)

    width, height = image_with_grid.size

    # Draw vertical lines
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=thickness)

    # Draw horizontal lines
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill=grid_color, width=thickness)

    return image_with_grid

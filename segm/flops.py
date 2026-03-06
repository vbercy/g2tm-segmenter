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
from typing import Tuple
import warnings

import click
from tqdm import tqdm
import numpy as np
from fvcore.nn import FlopCountAnalysis

import torch
from torch import nn
from torch.utils.data import Dataset

from segm.inference_utils import get_dataset_inference_path, dataset_prepare
from segm.model.factory import load_model
from segm.model.utils import padding
from segm.data.utils import STATS

import g2tm


warnings.filterwarnings("ignore")

TwoArrays = Tuple[np.array, np.array]


@torch.no_grad()
def compute_flops_per_image(model: nn.Module, validation_loader: Dataset,
                            device: str) -> np.array:
    """Average FLOPs made by the model for one forward-pass.

    This function computes the average number of floating-point operations
    (FLOPs) made by one model inference using the fvcore library tools.

    Args:
        model (nn.Module): PyTorch model.
        validation_loader (Dataset): PyTorch dataloader.
        device (str): Device to run the inference.

    Returns:
        mean_gflops (np.array): Mean GFLOPs for one inference.
    """
    gflops = []
    for image in tqdm(validation_loader, position=0, leave=False):
        image = image.to(device)
        flops = FlopCountAnalysis(model, image)
        flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        gflops.append(flops.total() / 1e9)

    return np.mean(gflops)


def compute_flops_per_image_per_module(model: nn.Module,
                                       validation_loader: Dataset,
                                       device: str) -> TwoArrays:
    """Average FLOPs made by the model encoder and decoder for one
    forward-pass.

    This function computes the average number of floating-point operations
    made by the encoder and the decoder of the model separatly, using the
    fvcore library tools.

    Args:
        model (nn.Module): PyTorch model.
        validation_loader (Dataset): PyTorch dataloader.
        device (str): Device to run the inference.

    Returns:
        mean_enc_gflops (np.array): Mean GFLOPs for the encoder.
        mean_dec_gflops (np.array): Mean GFLOPs for the decoder.
    """
    enc_gflops = []
    dec_gflops = []
    for image in tqdm(validation_loader, position=0, leave=False):
        image = image.to(device)
        image = padding(image, model.patch_size)
        h, w = image.size(2), image.size(3)

        encoded = model.encoder(image, return_features=True)

        enc_flops = FlopCountAnalysis(model.encoder, image)
        (enc_flops
         .unsupported_ops_warnings(False)
         .uncalled_modules_warnings(False))
        enc_gflops.append(enc_flops.total() / 1e9)

        if model.token_reduction:
            model.encoder.info["size"] = model.encoder.info["size"][:, 1:]
            if model.encoder.info["mask"] is not None:
                model.encoder.info["mask"] = (
                    model.encoder.info["mask"][:, 1:]
                )

        dec_flops = FlopCountAnalysis(model.decoder, (encoded[:, 1:], (h, w),
                                                      model.token_reduction))
        (dec_flops
         .unsupported_ops_warnings(False)
         .uncalled_modules_warnings(False))
        dec_gflops.append(dec_flops.total() / 1e9)

    return np.mean(enc_gflops), np.mean(dec_gflops)


@click.command()
@click.argument("model_path", type=str)
@click.argument("dataset_name", type=str)
@click.option("--batch-size", type=int, default=1)
@click.option("--patch-type", default="pure", type=str)
@click.option("--selected-layer", default=1, type=int)
@click.option("--threshold", default=0.88, type=float)
@click.option("--prop-attn/--no-prop-attn", default=False, is_flag=True)
@click.option("--iprop-attn/--no-iprop-attn", default=False, is_flag=True)
@click.option("--enc-dec/--no-enc-dec", default=False, is_flag=True)
def main(model_path: str, dataset_name: str, batch_size: int, patch_type: str,
         selected_layer: int, threshold: float, prop_attn: bool,
         iprop_attn: bool, enc_dec: bool):
    """Compute the number of floating-point operations made by the model.

    Args:
        model_path (str): Path to PyTorch model.
        dataset_name (str): Name of the dataset to use.
        patch_type (str): Token reduction method (pure => no reduction).
        selected_layer (int): Layer to apply token reduction (1-based).
        threshold (float): Threshold parameter for G2TM.
        prop_attn (bool): Whether to apply Proportional Attention.
        iprop_attn (bool): Whether to apply Inverse Proportional Attention.
        enc_dec (bool): Whether to compute GFLOPs for encoder and decoder
            separatly.
    """

    device = 'cuda:0'
    root_dir = os.getenv('DATASET')

    dataset_path, dataset_txt_path = get_dataset_inference_path(dataset_name,
                                                                root_dir)

    model, variant = load_model(model_path)
    input_size = variant['dataset_kwargs']['crop_size']
    normalization = variant["dataset_kwargs"]["normalization"]
    stats = STATS[normalization]

    if patch_type == "graph":
        g2tm.graph_segmenter_patch(model, selected_layer, threshold,
                                   prop_attn, iprop_attn)

    model.eval()
    model.to(device)

    validation_loader = dataset_prepare(dataset_path, dataset_txt_path, stats,
                                        batch_size, input_size)
    if enc_dec:
        enc_gflops, dec_gflops = compute_flops_per_image_per_module(
            model, validation_loader, device
        )
        print(f'Encoder GFlops: {enc_gflops:0.3f}', flush=True)
        print(f'Decoder GFlops: {dec_gflops:0.3f}', flush=True)

    gflops = compute_flops_per_image(model, validation_loader, device)
    print(f'GFlops: {gflops:0.3f}', flush=True)


if __name__ == "__main__":
    main()

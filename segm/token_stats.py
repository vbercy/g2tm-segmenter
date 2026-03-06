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


import os
import click
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

import segm.utils.torch as ptu
from segm.inference_utils import get_dataset_inference_path, dataset_prepare
from segm.data.utils import STATS
from segm.model.factory import load_model

import g2tm


def fill_stats(li: list, q: float = 0.9) -> dict:
    """Compute statistics from list of values.
    Args:
        li (list): List of values.
        q (float): Quantile.

    Returns:
        dict: Dictionnary containing the statistics.
    """
    return {
            "mean": np.mean(li),
            "median": np.median(li),
            f"q{int(100-q*100)}": np.quantile(li, 1-q),
            f"q{int(q*100)}": np.quantile(li, q)
        }


def pretty_print_dict(d: dict, unit: str = "", indent: int = 0):
    """Pretty print dictionnary

    Args:
        d (dict): Dictionnary to print.
        unit (str): Unit for numerical value.
        indent (int): Number of indent before printing values.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print('\t' * indent + str(key))
            pretty_print_dict(value, unit, indent + 1)
        else:
            print('\t' * indent + f"{key}: {value:.3f} {unit}")


@torch.no_grad()
def print_fusion_stats(model: nn.Module, validation_loader: DataLoader,
                       layer_id: int, patch_type: str):
    """Getting fusion statistics for a certain encoder layer ID.

    This function computes statistics on the token sequence (mean, median
    and quantiles number of tokens and token size) across the entire
    dataset. The layer ID should be high enough, so that the data has
    passed through a token reduction block and so that the encoder's
    "size" key of its "_info" dictionnary attribute is not None.

    Args:
        model (nn.Module): PyTorch model.
        validation_loader (DataLoader): PyTorch dataloader.
        layer_id (int): Layer to compute token statistics (0-based).
        patch_type (str): Token reduction method (pure => no reduction).
    """
    print(f"Getting token fusion stats from encoder layer ID {layer_id}.")
    token_number = []
    token_size = []
    print_warning = False

    for image in tqdm(validation_loader, position=0, leave=False):
        # Run the encoder forward until we reach the desired layer
        image = image.to(ptu.device)
        attn_map = model.get_attention_map_enc(image.to(ptu.device), layer_id)
        s = model.encoder.info["size"]

        if s is None:
            print_warning = True
            n = attn_map.size(2) - 1
            s = np.ones((1, n), dtype=float)

        if patch_type == "graph":
            token_number.append(s.shape[1])
            token_size += list(filter((1).__ne__, s[0].tolist()))

    if print_warning:
        print("WARNING: token size tensor is None, either the module has not"
              " find any fusion in this image, or no fusion module has been"
              " trigered. In this case, consider increasing the layer ID.")

    out = {"Number of tokens": {}, "Size of tokens (!=1)": {}}
    out["Number of tokens"] = fill_stats(token_number, 0.95)
    out["Size of tokens (!=1)"] = fill_stats(token_size)
    pretty_print_dict(out)


@click.command()
@click.argument("model_path", type=str)
@click.argument("dataset_name", type=str)
@click.option("--layer-id", default=0, type=int)
@click.option("--patch-type", default="pure", type=str)
@click.option("--selected-layer", default=1, type=int)
@click.option("--threshold", default=0.88, type=float)
@click.option("--prop-attn/--no-prop-attn", default=False, is_flag=True)
@click.option("--iprop-attn/--no-iprop-attn", default=False, is_flag=True)
def main(model_path, dataset_name, layer_id, patch_type,
         selected_layer, threshold, prop_attn, iprop_attn):
    """Compute the token statistics of the model.

    Args:
        model_path (str): Path to PyTorch model.
        dataset_name (str): Name of the dataset to use.
        layer_id (int): Layer to compute token statistics (0-based).
        patch_type (str): Token reduction method (pure => no reduction).
        selected_layer (int): Layer to apply token reduction (1-based).
        threshold (float): Threshold parameter for G2TM.
        prop_attn (bool): Whether to apply Proportional Attention.
        iprop_attn (bool): Whether to apply Inverse Proportional Attention.
    """
    ptu.set_gpu_mode(True)

    batch_size = 1
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
    else:
        raise ValueError("No token reduction applied. This script has no"
                         "interest for vanilla models.")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(ptu.device)

    validation_loader = dataset_prepare(dataset_path, dataset_txt_path, stats,
                                        batch_size, input_size)
    print_fusion_stats(model, validation_loader, layer_id, patch_type)


if __name__ == "__main__":
    main()

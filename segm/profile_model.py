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
from pathlib import Path
import warnings

import click

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

from segm.inference_utils import get_dataset_inference_path, dataset_prepare
from segm.model.factory import load_model
from segm.data.utils import STATS

import g2tm


warnings.filterwarnings("ignore")


@torch.no_grad()
def profile_model(model: nn.Module, validation_loader: DataLoader,
                  device: str | torch.device) -> profile:
    """Profile PyTorch model activity on CPU and GPU.

    This function profiles the activity of a model during inference,
    using PyTorch built-in tools. It provides both CPU and GPU profiling.

    Args:
        model (nn.Module): PyTorch model.
        validation_loader (DataLoader): PyTorch dataloader.
        device (str|torch.device): Device to run the model.

    Returns:
        prof (profile): PyTorch profiler with model activity stored.
    """
    torch.cuda.empty_cache()
    if not isinstance(device, torch.device):
        device = torch.device(device)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    torch.cuda.synchronize()
    with torch.no_grad():
        for i, image in enumerate(validation_loader):
            # warmup iterations
            if i < 3:
                model(image.to(device))

            # profiling iteration
            elif i == 3:
                with (profile(activities=activities, record_shapes=True)
                      as prof):
                    with record_function("model_inference"):
                        model(image.to(device))

                n_tokens = (model.encoder.info["source"].size(1)
                            if model.token_reduction
                            else model.encoder.patch_embed.num_patches)
                print(f"Number of tokens after reduction: {n_tokens}.")
                break

    return prof


@click.command()
@click.argument("model_path", type=str)
@click.argument("dataset_name", type=str)
@click.option("--patch-type", default="pure", type=str)
@click.option("--selected-layer", default=1, type=int)
@click.option("--threshold", default=0.88, type=float)
@click.option("--prop-attn/--no-prop-attn", default=False, is_flag=True)
@click.option("--iprop-attn/--no-iprop-attn", default=False, is_flag=True)
def main(model_path: str, dataset_name: str, patch_type: str,
         selected_layer: int, threshold: float, prop_attn: bool,
         iprop_attn: bool):
    """Profiles model activity during inference on CPU and GPU.

    Args:
        model_path (str): Path to PyTorch model.
        dataset_name (str): Name of the dataset to use.
        patch_type (str): Token reduction method (pure => no reduction).
        selected_layer (int): Layer to apply token reduction (1-based).
        threshold (float): Threshold parameter for G2TM.
        prop_attn (bool): Whether to apply Proportional Attention.
        iprop_attn (bool): Whether to apply Inverse Proportional Attention.
    """

    device = 'cuda:0'

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

    model.eval()
    model.to(device)

    validation_loader = dataset_prepare(dataset_path, dataset_txt_path, stats,
                                        batch_size, input_size, False)
    prof = profile_model(model, validation_loader, device)

    save_folder = Path(model_path).parent
    prof.export_chrome_trace(str(save_folder / "trace.json"))
    print(prof.key_averages().table(sort_by='self_cuda_time_total',
                                    row_limit=10))
    print(prof.key_averages().table(sort_by='self_cpu_time_total',
                                    row_limit=10))


if __name__ == "__main__":
    main()

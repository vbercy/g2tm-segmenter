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


import warnings

import click
import torch

from segm.engine import evaluate
from segm.data.factory import create_dataset
from segm.model.factory import load_model
from segm.utils import distributed
import segm.utils.torch as ptu

import g2tm


warnings.filterwarnings("ignore")


@torch.no_grad()
@click.command()
@click.argument("model_path", type=str)
@click.option("--patch-type", default="pure", type=str)
@click.option("--selected-layer", default=1, type=int)
@click.option("--threshold", default=0.88, type=float)
@click.option("--prop-attn/--no-prop-attn", default=False, is_flag=True)
@click.option("--iprop-attn/--no-iprop-attn", default=False, is_flag=True)
def main(model_path: str, patch_type: str, selected_layer: int,
         threshold: float, prop_attn: bool, iprop_attn: bool):
    """Compute the mIoU score of the model.

    Args:
        model_path (str): Path to PyTorch model.
        patch_type (str): Token reduction method (pure => no reduction).
        selected_layer (int): Layer to apply token reduction (1-based).
        threshold (float): Threshold parameter for G2TM.
        prop_attn (bool): Whether to apply Proportional Attention.
        iprop_attn (bool): Whether to apply Inverse Proportional Attention.
    """
    ptu.set_gpu_mode(True)
    distributed.init_process()

    model, variant = load_model(model_path)
    dataset_kwargs = variant['dataset_kwargs']
    dataset_kwargs["batch_size"] = 1
    dataset_kwargs["split"] = "val"
    dataset_kwargs["crop"] = False
    validation_loader = create_dataset(dataset_kwargs)

    if patch_type == "graph":
        g2tm.graph_segmenter_patch(model, selected_layer, threshold,
                                   prop_attn, iprop_attn)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(ptu.device)

    amp_autocast = ptu.get_autocast(variant["amp"])

    val_seg_gt = validation_loader.dataset.get_gt_seg_maps()
    eval_logger = evaluate(model, validation_loader, val_seg_gt,
                           variant["inference_kwargs"]["window_size"],
                           variant["inference_kwargs"]["window_stride"],
                           amp_autocast)

    print("Metrics:", eval_logger, flush=True)
    print(str(eval_logger.mean_iou).split(" ", maxsplit=1)[0])
    print("")


if __name__ == "__main__":
    main()

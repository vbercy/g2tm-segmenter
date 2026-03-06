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


import click

import torch

import segm.utils.torch as ptu
from segm.utils.logger import MetricLogger
from segm.model.factory import create_vit
from segm.data.factory import create_dataset
from segm.data.utils import STATS
from segm.metrics import accuracy
from segm import config


def compute_labels(model, batch):
    """ Compute ImageNet accuracy metrics.
    """
    im = batch["im"]
    target = batch["target"]

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model.forward(im)
    acc1, acc5 = accuracy(output, target, topk=(1, 5))

    return acc1.item(), acc5.item()


def eval_dataset(model, dataset_kwargs):
    """ Evaluate the dataset on ImageNet accuracy metrics.
    """
    db = create_dataset(dataset_kwargs)
    print_freq = 20
    header = ""
    logger = MetricLogger(delimiter="  ")

    for batch in logger.log_every(db, print_freq, header):
        for k, v in batch.items():
            batch[k] = v.to(ptu.device)
        acc1, acc5 = compute_labels(model, batch)
        batch_size = batch["im"].size(0)
        logger.update(acc1=acc1, n=batch_size)
        logger.update(acc5=acc5, n=batch_size)
    print(f"Imagenet accuracy: {logger}")


@click.command()
@click.argument("backbone", type=str)
@click.option("--imagenet-dir", type=str)
@click.option("-bs", "--batch-size", default=32, type=int)
@click.option("-nw", "--num-workers", default=10, type=int)
@click.option("-gpu", "--gpu/--no-gpu", default=True, is_flag=True)
def main(backbone, imagenet_dir, batch_size, num_workers, gpu):
    """ Compute the model's ImageNet accuracy scores.
    """
    ptu.set_gpu_mode(gpu)
    cfg = config.load_config()
    cfg = cfg["model"][backbone]
    cfg["backbone"] = backbone
    cfg["image_size"] = (cfg["image_size"], cfg["image_size"])

    dataset_kwargs = {
        "dataset": 'imagenet',
        "root_dir": imagenet_dir,
        "image_size": cfg['image_size'],
        "crop_size": cfg['image_size'],
        "patch_size": cfg['patch_size'],
        "batch_size": batch_size,
        "num_workers": num_workers,
        "split": 'val',
        "normalization": STATS[cfg["normalization"]],
    }

    model = create_vit(cfg)
    model.to(ptu.device)
    model.eval()
    eval_dataset(model, dataset_kwargs)


if __name__ == "__main__":
    main()

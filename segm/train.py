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

# Modifications based on code from Robin Strudel et al. (Segmenter)

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


import json
import argparse
from pathlib import Path
import yaml
import torch
import click
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from timm.utils import NativeScaler

from segm.utils import distributed
import segm.utils.torch as ptu
from segm import config
from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.data.factory import create_dataset
from segm.model.utils import num_params
from segm.utils.distributed import sync_model
from segm.engine import train_one_epoch, evaluate

import g2tm


@click.command(help="")
@click.option("--log-dir", type=str, help="logging directory")
@click.option("--dataset", type=str)
@click.option("--im-size", default=None, type=int, help="dataset resize size")
@click.option("--crop-size", default=None, type=int)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="", type=str)
@click.option("--decoder", default="", type=str)
@click.option("--optimizer", default="sgd", type=str)
@click.option("--scheduler", default="polynomial", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.1, type=float)
@click.option("--batch-size", default=None, type=int)
@click.option("--epochs", default=None, type=int)
@click.option("-lr", "--learning-rate", default=None, type=float)
@click.option("-min-lr", "--minlearning-rate", default=1e-5, type=float)
@click.option("--warmup-epochs", default=0, type=int)
@click.option("--start-factor", default=1., type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--eval-freq", default=None, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)
@click.option("--resume/--no-resume", default=True, is_flag=True)
@click.option("--patch-type", default="pure", type=str)
@click.option("--selected-layer", default=1, type=int)
@click.option("--threshold", default=0.88, type=float)
@click.option("--start-thresh", default=0.95, type=float)
@click.option("--curric-warmup", default=64, type=int)
@click.option("--curric-period", default=10, type=int)
@click.option("--curric-thresh/--no-curric-thresh",
              default=False, is_flag=True)
@click.option("--prop-attn/--no-prop-attn", default=False, is_flag=True)
@click.option("--iprop-attn/--no-iprop-attn", default=False, is_flag=True)
@click.option("--n-cycles", default=None, type=int)
def main(log_dir, dataset, im_size, crop_size, window_size, window_stride,
         backbone, decoder, optimizer, scheduler, weight_decay, dropout,
         drop_path, batch_size, epochs, learning_rate, minlearning_rate,
         warmup_epochs, start_factor, normalization, eval_freq, amp, resume,
         patch_type, selected_layer, threshold, start_thresh, curric_warmup,
         curric_period, curric_thresh, prop_attn, iprop_attn, n_cycles):
    """Model training with or without token reduction.

    Args:
        log_dir (str): Directory to save logs and models.
        dataset (str): Name of the dataset.
        im_size (int): Size of the images in the dataset.
        crop_size (int): Size of the image crop.
        window_size (int): Size of the sliding window during evaluation.
        window_stride (int): Step of the sliding window during evaluation.
        backbone (str): Architecture of the backbone/encoder.
        decoder (str): Architecture of the decoder.
        optimizer (str): Optimizer to use.
        scheduler (str): Learning rate scheduler to use.
        weight_decay (float): Weigth decay value.
        dropout (float): dropout probability applied to attention weights.
        drop_path (float): Probability of dropping the residual path.
        batch_size (int): Number of images per batch.
        epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate value (without scheduler).
        minlearning_rate (float): Minimum value for the learning rate.
        warmup_epochs (int): Number of warmup epochs.
        start_factor (float): Factor applied to the learning rate at start.
        normalization (str): What normalization statistics to use ('vit' or
            'deit').
        eval_freq (int): Number of epochs between two evaluation steps.
        amp (bool): Whether to activate AMP Autocast.
        resume (bool): Whether to resume training from the provided directory.
        patch_type (str): Token reduction method (pure => no reduction).
        selected_layer (int): Layer to apply token reduction (1-based).
        threshold (float): Threshold parameter for G2TM.
        start_thresh (float): Starting value of the threshold if curriculum.
        curric_warmup (int): Number of epoch for which curriculum is disabled.
        curric_period (int): Number of epochs of each curriculum step.
        curric_thresh (bool): Whether to activate threshold curriculum.
        prop_attn (bool): Whether to apply Proportional Attention.
        iprop_attn (bool): Whether to apply Inverse Proportional Attention.
        n_cycles (int): Number of cycles for the cosine scheduler.

    """
    # start distributed mode
    ptu.set_gpu_mode(True)
    distributed.init_process()

    # set up configuration
    cfg = config.load_config()
    # print(cfg["model"])

    model_cfg = cfg["model"][backbone]

    dataset_cfg = cfg["dataset"][dataset]
    if "mask_transformer" in decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][decoder]

    # model config
    if not im_size:
        im_size = dataset_cfg["im_size"]
    if not crop_size:
        crop_size = dataset_cfg.get("crop_size", im_size)
    if not window_size:
        window_size = dataset_cfg.get("window_size", im_size)
    if not window_stride:
        window_stride = dataset_cfg.get("window_stride", im_size)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    # dataset config
    world_batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    lr = dataset_cfg["learning_rate"]
    if batch_size:
        world_batch_size = batch_size
    if epochs:
        num_epochs = epochs
    if learning_rate:
        lr = learning_rate
    if eval_freq is None:
        eval_freq = dataset_cfg.get("eval_freq", 1)

    if normalization:
        model_cfg["normalization"] = normalization

    if patch_type != "graph":
        threshold = None
        start_thresh = None
        selected_layer = None

    if not curric_thresh:
        start_thresh = threshold

    if scheduler == "cosine":
        assert num_epochs % n_cycles == 0
        epochs = num_epochs // n_cycles
    else:
        epochs = num_epochs

    # experiment config
    batch_size = world_batch_size // ptu.world_size
    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=resume,
        dataset_kwargs=dict(
            dataset=dataset,
            image_size=im_size,
            crop_size=crop_size,
            batch_size=batch_size,
            normalization=model_cfg["normalization"],
            split="train",
            num_workers=10,
            patch_type=patch_type,
            threshold=threshold,
            start_threshold=start_thresh,
            curriculum_start=curric_warmup,
            curriculum_period=curric_period,
            selected_layer=selected_layer,
        ),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            start_epoch=0,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
        ),
        optimizer_kwargs=dict(
            opt=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            clip_grad=None,
            sched=scheduler,
            epochs=epochs,
            min_lr=minlearning_rate,
            poly_power=0.9,
            poly_step_size=1,
            warmup_epochs=warmup_epochs,
            start_factor=start_factor,
            # test Cosine Scheduler
            t_initial=num_epochs // 64,
            decay_rate=0.8,
            warmup_lr=start_factor * lr,
        ),
        net_kwargs=model_cfg,
        amp=amp,
        log_dir=log_dir,
        inference_kwargs=dict(
            im_size=im_size,
            window_size=window_size,
            window_stride=window_stride,
        ),
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = log_dir / "checkpoint.pth"
    best_checkpoint_path = log_dir / "best_checkpoint.pth"

    # dataset

    dataset_kwargs = variant["dataset_kwargs"]

    if dataset_kwargs["dataset"] == 'ade20k_large':
        dataset_kwargs["dataset"] = 'ade20k'
    elif dataset_kwargs["dataset"] == 'cityscapes_large':
        dataset_kwargs["dataset"] = 'cityscapes'

    train_loader = create_dataset(dataset_kwargs)
    val_kwargs = dataset_kwargs.copy()
    val_kwargs["split"] = "val"
    val_kwargs["batch_size"] = 1
    val_kwargs["crop"] = False
    val_loader = create_dataset(val_kwargs)
    n_cls = train_loader.unwrapped.n_cls

    # model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["n_cls"] = n_cls
    model = create_segmenter(net_kwargs)

    if patch_type == "graph":
        g2tm.graph_segmenter_patch(model, selected_layer, start_thresh,
                                   prop_attn, iprop_attn)

    model.to(ptu.device)
    optimizer_kwargs = variant["optimizer_kwargs"]
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v
    optimizer = create_optimizer(opt_args, model)
    lr_scheduler = create_scheduler(opt_args, optimizer)
    amp_autocast = ptu.get_autocast(amp)
    loss_scaler = None
    if amp:
        loss_scaler = NativeScaler()

    variant["algorithm_kwargs"]["start_epoch"] = 0
    # resume and fine tune
    if resume and checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = ptu.load_checkpoint(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])

        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            variant["algorithm_kwargs"]["start_epoch"] = checkpoint["epoch"]+1

        if loss_scaler and "loss_scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["loss_scaler"])

        if lr_scheduler and "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        if threshold and "threshold" in checkpoint:
            model.encoder.threshold = checkpoint["threshold"]
            model.encoder.info["threshold"] = checkpoint["threshold"]

    else:
        sync_model(log_dir, model)

    if ptu.distributed:
        ddp_kwargs = {"find_unused_parameters": True}
        ddp_device_ids = ptu.get_ddp_device_ids()
        if ddp_device_ids is not None:
            ddp_kwargs["device_ids"] = ddp_device_ids
            ddp_kwargs["output_device"] = ddp_device_ids[0]
        model = DDP(model, **ddp_kwargs)

    # save config
    variant_str = yaml.dump(variant)
    print(f"Configuration:\n{variant_str}")
    variant["net_kwargs"] = net_kwargs
    variant["dataset_kwargs"] = dataset_kwargs
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "variant.yml", "w", encoding='utf-8') as f:
        f.write(variant_str)

    # train
    start_epoch = variant["algorithm_kwargs"]["start_epoch"]
    num_epochs = variant["algorithm_kwargs"]["num_epochs"]
    eval_freq = variant["algorithm_kwargs"]["eval_freq"]

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    val_seg_gt = val_loader.dataset.get_gt_seg_maps()

    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Val dataset length: {len(val_loader.dataset)}")
    print(f"Encoder parameters: {num_params(model_without_ddp.encoder)}")
    print(f"Decoder parameters: {num_params(model_without_ddp.decoder)}")

    writer = SummaryWriter(log_dir / "tensorboard")

    best_checkpoint = 0
    for epoch in range(start_epoch, num_epochs):
        writer.add_scalar("epoch", epoch, epoch, new_style=True)

        if curric_thresh:
            if (epoch > curric_warmup - 1 and
                    (epoch - curric_warmup) % curric_period == 0 and
                    model_without_ddp.encoder.threshold > threshold):
                model_without_ddp.encoder.threshold = (
                    round(model_without_ddp.encoder.threshold-0.01, 2)
                )
                model_without_ddp.encoder.info["threshold"] = (
                    round(model_without_ddp.encoder.info["threshold"]-0.01, 2)
                )
                best_checkpoint = 0.0
            writer.add_scalar("threshold",
                              model_without_ddp.encoder.info["threshold"],
                              epoch, new_style=True)

        # train for one epoch
        train_logger = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
            loss_scaler,
            writer
        )

        # save checkpoint
        if ptu.dist_rank == 0:
            snapshot = dict(
                model=model_without_ddp.state_dict(),
                optimizer=optimizer.state_dict(),
                n_cls=model_without_ddp.n_cls,
                lr_scheduler=lr_scheduler.state_dict(),
            )
            if loss_scaler is not None:
                snapshot["loss_scaler"] = loss_scaler.state_dict()
            snapshot["epoch"] = epoch
            if patch_type != "pure":
                snapshot["threshold"] = model_without_ddp.encoder.threshold
            torch.save(snapshot, checkpoint_path)

        # evaluate
        eval_epoch = epoch % eval_freq == 0 or epoch == num_epochs - 1
        if eval_epoch:
            eval_logger = evaluate(
                model,
                val_loader,
                val_seg_gt,
                window_size,
                window_stride,
                amp_autocast,
            )

            writer.add_scalar("val_pixel_acc",
                              eval_logger.pixel_accuracy.median,
                              epoch, new_style=True)
            writer.add_scalar("val_mean_acc",
                              eval_logger.mean_accuracy.median,
                              epoch, new_style=True)
            writer.add_scalar("val_mean_iou",
                              eval_logger.mean_iou.median,
                              epoch, new_style=True)

            print(f"Stats [{epoch}]:", eval_logger, flush=True)
            print(str(eval_logger.mean_iou).split(" ", maxsplit=1)[0])
            print("")

            curr_checkpoint = (str(eval_logger.mean_iou)
                                .split(" ", maxsplit=1)[0])
            if float(curr_checkpoint) > best_checkpoint:
                print("saving the best checkpoint ...")
                best_checkpoint = float(curr_checkpoint)
                best_checkpoint_name = (
                    "best_checkpoint_" + str(best_checkpoint) + ".pth"
                )
                best_checkpoint_path = log_dir / best_checkpoint_name
                torch.save(snapshot, best_checkpoint_path)

        # log stats
        if ptu.dist_rank == 0:
            train_stats = {
                k: meter.global_avg for k,
                meter in train_logger.meters.items()
            }
            val_stats = {}
            if eval_epoch:
                val_stats = {
                    k: meter.global_avg for k,
                    meter in eval_logger.meters.items()
                }

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "num_updates": (epoch + 1) * len(train_loader),
                "threshold": (model_without_ddp.encoder.threshold
                              if (patch_type != "pure") else None),
            }

            with open(log_dir / "log.txt", "a", encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + "\n")

    distributed.barrier()
    distributed.destroy_process()
    print(f"Finished running DDP training on rank {ptu.dist_rank}.")


if __name__ == "__main__":
    main()

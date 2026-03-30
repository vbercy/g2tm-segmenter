# Training commands

Below, we provide the training commands for training Segmenter + G2TM.

The following datasets can be loaded using the `dataset` option:
* [ADE20K](https://ade20k.csail.mit.edu/)
* [Cityscapes](https://www.cityscapes-dataset.com/)
* [Pascal-Context](https://cs.stanford.edu/~roozbeh/pascal-context/)

Different backbone sizes are available for Segmenter using the `backbone` option:
* Segmenter with ViT-T (Seg-T): `vit_tiny_patch16_384`
* Segmenter with ViT-S (Seg-S): `vit_small_patch16_384`
* Segmenter with ViT-B (Seg-B): `vit_base_patch16_384`
* Segmenter with ViT-L (Seg-L): `vit_large_patch16_384`

For more configuration options, see [segm/config.yml](segm/config.yml) and [segm/train.py](segm/train.py).

Do not forget to define the `DATASET` environment variable:

```bash
export DATASET=/database2/ADE20K
```

## Single GPU on ADE20K

The following commands enable single-GPU training with different ViT backbone sizes.

Segmenter + G2TM at layer 2 with threshold 0.88, and ViT-T backbone:

```bash
python ./segm/train.py --log-dir <tiny_model_dir> \
                       --dataset ade20k \
                       --backbone vit_tiny_patch16_384 \
                       --decoder mask_transformer \
                       --patch-type graph \
                       --selected-layer 2 \
                       --threshold 0.88 \
                       --batch-size 8
```

Segmenter + G2TM at layer 2 with threshold 0.88, and ViT-S backbone:

```bash
python ./segm/train.py --log-dir <small_model_dir> \
                       --dataset ade20k \
                       --backbone vit_small_patch16_384 \
                       --decoder mask_transformer \
                       --patch-type graph \
                       --selected-layer 2 \
                       --threshold 0.88 \
                       --batch-size 8
```

Segmenter + G2TM at layer 2 with threshold 0.88, and ViT-B backbone:

```bash
python ./segm/train.py --log-dir <base_model_dir> \
                       --dataset ade20k \
                       --backbone vit_base_patch16_384 \
                       --decoder mask_transformer \
                       --patch-type graph \
                       --selected-layer 2 \
                       --threshold 0.88 \
                       --batch-size 8
```

Segmenter + G2TM at layer 2 with threshold 0.88, and ViT-L backbone:

```bash
python ./segm/train.py --log-dir <large_model_dir> \
                       --dataset ade20k \
                       --backbone vit_large_patch16_384 \
                       --decoder mask_transformer \
                       --patch-type graph \
                       --selected-layer 2 \
                       --threshold 0.88 \
                       --batch-size 8
```

## Multi GPU on Cityscapes 

The following command uses torch.distributed.run to enable multi-GPU training

Segmenter + G2TM at layer 2 with threshold 0.94, ViT-L backbone, with 2 GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run \
       --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=127.0.0.1:24900 \
       ./segm/train.py \
       --log-dir <cityscapes_model_dir> \
       --dataset cityscapes \
       --backbone vit_large_patch16_384 \
       --decoder mask_transformer \
       --patch-type graph \
       --selected-layer 2 \
       --threshold 0.94 \
       --batch-size 8
```

**NOTE:** The next examples are given with single-GPU configurations, but they can be applied in multi-GPU context.

## Image resolution

The following command allows you to change the input image resolution, which directly impacts the number of tokens in the sequence. Like the paper, we set the image resolution of Cityscapes to 1024×1024, instead of the default 728×728.

Segmenter + G2TM at layer 2 with threshold 0.94, and ViT-S backbone:

```bash
python ./segm/train.py --log-dir <citylarge_model_dir> \
                       --dataset cityscapes_large \
                       --backbone vit_small_patch16_384 \
                       --decoder mask_transformer \
                       --patch-type graph \
                       --selected-layer 2 \
                       --threshold 0.94 \
                       --batch-size 8
```

## (Inverse) Proportional Attention

The following commands enable Proportional Attention and Inverse Proportional Attention respectively, for all attention layers that follows the merging module.

Segmenter + G2TM at layer 2 with threshold 0.88, with Proportional Attention and ViT-S backbone:

```bash
python ./segm/train.py --log-dir <pa_model_dir> \
                       --dataset ade20k \
                       --backbone vit_small_patch16_384 \
                       --decoder mask_transformer \
                       --patch-type graph \
                       --selected-layer 2 \
                       --threshold 0.94 \
                       --batch-size 8 \
                       --prop-attn
```

Segmenter + G2TM at layer 2 with threshold 0.88, with Inverse Proportional Attention and ViT-S backbone:

```bash
python ./segm/train.py --log-dir <ipa_model_dir> \
                       --dataset ade20k \
                       --backbone vit_small_patch16_384 \
                       --decoder mask_transformer \
                       --patch-type graph \
                       --selected-layer 2 \
                       --threshold 0.88 \
                       --batch-size 8 \
                       --iprop-attn
```

## Curriculum

The following command enables the threshold to vary during the training, by activating the `curric-tresh` option. You can tweak the curriculum using the options below:
* `threshold`: The final threshold value, at the end of training
* `start-thresh`: The starting threshold value
* `curric-start`: The epoch to start the curriculum (from 1)
* `curric-period`: The number of epochs for each curriculum step

**NOTE:** Between two curriculum steps, the threshold value is decreased by 0.01. Hence, you have to tweak the options above carefully to get the expected final value.

Segmenter + G2TM at layer 2 with threshold 0.88, with curriculum and ViT-S backbone:

```bash
python ./segm/train.py --log-dir <curric_model_dir> \
                       --dataset ade20k \
                       --backbone vit_small_patch16_384 \
                       --decoder mask_transformer \
                       --patch-type graph \
                       --selected-layer 2 \
                       --threshold 0.88 \
                       --batch-size 8 \
                       --curric-thresh \
                       --start-thresh 0.95 \
                       --curric-start 64 \
                       --curric-period 10
```

Here, the model is trained with a starting threshold value of 0.95 during 64 epochs. From the 65th epoch, the threshold value is decreased by 0.01 every 10 epochs.
# MIT License

# Copyright (c) 2021 Robin Strudel
# Copyright (c) INRIA

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# dataset settings
dataset_type = "ADE20KDataset"
data_root = "/database2/ADE20K"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 512)
max_ratio = 4
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="RandomResize", scale=(512 * max_ratio, 512),
         ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=dict(img=0, seg=255)),
    dict(type="PackSegInputs"),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', scale=(512 * max_ratio, 512), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', scale=(512 * max_ratio, 512), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PackSegInputs')
]
fps_val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', scale=(512 * max_ratio, 512), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PackSegInputs')
]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale=(512 * max_ratio, 512), keep_ratio=True)],
            [
                dict(type='RandomFlip', prob=0.),
                dict(type='RandomFlip', prob=1.)
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images/training",
                         seg_map_path="annotations/training"),
        pipeline=train_pipeline,
    ),
    trainval=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=["images/training", "images/validation"],
            seg_map_path=["annotations/training", "annotations/validation"]
        ),
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images/validation",
                         seg_map_path="annotations/validation"),
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images/validation",
                         seg_map_path="annotations/validation"),
        pipeline=test_pipeline,
    ),
    fps_val=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="images/validation",
                         seg_map_path=""),
        pipeline=fps_val_pipeline,
    ),
)

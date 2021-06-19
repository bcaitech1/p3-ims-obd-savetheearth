# Model
# model settings


classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='DetectoRS_ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),

    neck=dict(
        type='RFP',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            # pretrained='torchvision://resnet50',
            style='pytorch')),

    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 48]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), # veri-focal loss 추가
        loss_bbox=dict(
            type='SmoothL1Loss', beta= 1.0 / 9, loss_weight=1.0)),

    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64]),
        bbox_head=[
            dict(
                type='VFNetHead',
                num_classes=11,
                in_channels=256,
                stacked_convs=3,
                feat_channels=256,
                strides=[8, 16, 32, 64, 128],
                center_sampling=False,
                dcn_on_last_conv=True,
                use_atss=True,
                use_vfl=True,
                loss_cls=dict(
                    type='VarifocalLoss',
                    use_sigmoid=True,
                    alpha=0.75,
                    gamma=2.0,
                    iou_weighted=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
                loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
           dict(
                type='VFNetHead',
                num_classes=11,
                in_channels=256,
                stacked_convs=3,
                feat_channels=256,
                strides=[8, 16, 32, 64, 128],
                center_sampling=False,
                dcn_on_last_conv=True,
                use_atss=True,
                use_vfl=True,
                loss_cls=dict(
                    type='VarifocalLoss',
                    use_sigmoid=True,
                    alpha=0.75,
                    gamma=2.0,
                    iou_weighted=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
                loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
            dict(
                type='VFNetHead',
                num_classes=11,
                in_channels=256,
                stacked_convs=3,
                feat_channels=256,
                strides=[8, 16, 32, 64, 128],
                center_sampling=False,
                dcn_on_last_conv=True,
                use_atss=True,
                use_vfl=True,
                loss_cls=dict(
                    type='VarifocalLoss',
                    use_sigmoid=True,
                    alpha=0.75,
                    gamma=2.0,
                    iou_weighted=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
                loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            # nms=dict(type='nms', iou_threshold=0.5),
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),
            max_per_img=100)))


# Dataset
dataset_type = 'CocoDataset'
data_root = '/opt/ml/input/data/'

# custom dataset에 맞게 mean, std 조정
img_norm_cfg = dict(mean=[117.302, 112.092, 106.659], std=[53.759, 52.954, 55.223], to_rgb=True)

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=10,
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1.0),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=1.0),
        ],
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.3),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512,512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=[0.8, 0.9, 1.0, 1.1, 1.2],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'train_data4.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'valid_data4.json',
        img_prefix=data_root,
        pipeline=val_pipeline,
        samples_per_gpu=16), # Batch size of a single GPU used in validation
    test=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))  # Batch size of a single GPU used in test



# schedules
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Learning rate scheduler config used to register LrUpdater hook
lr_config = dict(
    policy='step', # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    warmup='linear', # The warmup policy, also support `exp` and `constant`.
    warmup_iters=500, # The number of iterations for warmup
    warmup_ratio=0.001,
    step=[8, 18, 32]) # Steps to decay the learning rate
runner = dict(type='EpochBasedRunner', max_epochs=50)
evaluation = dict(interval=1, metric='bbox')


# runtime
checkpoint_config = dict(interval=1)

# The logger used to record the training process.
log_config = dict(
    interval=25,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'

# load models as a pre-trained model from a given path. This will not resume training.
load_from ='./models/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth'

# Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
resume_from = None

# Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once.
# The workflow trains the model by 12 epochs according to the total_epochs.
# workflow 뭔지 조사
workflow = [('train', 1)]

# Directory to save the model checkpoints and logs for the current experiments.
work_dir = './work_dirs/detectoRS_r50_rfp_ver3_varifocalloss'
seed=2020
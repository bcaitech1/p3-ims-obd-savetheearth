_base_ = [
    '/opt/ml/code/mmdetection_trash/configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py',
    '../dataset.py',
    # '../../_base_/schedules/schedule_2x.py',
    # '../../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=11
        )
    )
)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

checkpoint_config = dict(max_keep_ckpts=3, interval=1)

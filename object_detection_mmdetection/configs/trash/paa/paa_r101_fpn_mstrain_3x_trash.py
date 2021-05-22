_base_ = './paa_r50_fpn_mstrain_3x_trash.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

_base_ = './paa_r50_fpn_1x_trash.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

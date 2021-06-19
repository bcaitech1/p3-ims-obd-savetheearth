import torch
# import cv2
import torch.nn as nn
import torch.nn.functional as F
import cv2

# from albumentations import Compose, Normalize
# from albumentations.pytorch.transforms import ToTensorV2

# import segmentation_models_pytorch as smp

# class SMP_FPN_effb4_ns(nn.Module):
#     def __init__(self, num_classes=12):
#         super(SMP_FPN_effb4_ns, self).__init__()
#         self.seg_model = smp.FPN(encoder_name="timm-efficientnet-b4",
#                                   encoder_weights="noisy-student",
#                                   in_channels=3,
#                                   classes=12)

#     def forward(self, x):
#         x = self.seg_model(x)
#         return x

# a = SMP_FPN_effb4_ns()
# print(a)
print(torch.tensor([1,1,1]))
image = cv2.imread('C:/Users/user/Desktop/p3-ims-obd-savetheearth/WEB_P3/p3_web/media/images/1.jpg')
print(image)
import segmentation_models_pytorch as smp
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

#seg 모델
class SMP_FPN_effb4_ns(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_FPN_effb4_ns, self).__init__()
        self.seg_model = smp.FPN(encoder_name="timm-efficientnet-b4",
                                  encoder_weights="noisy-student",
                                  in_channels=3,
                                  classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x

class SMP_DeepLabV3Plus_resnext101_32x4d(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_DeepLabV3Plus_resnext101_32x4d, self).__init__()
        self.seg_model = smp.DeepLabV3Plus(encoder_name="resnext101_32x4d",
                                          encoder_weights="ssl",
                                          in_channels=3,
                                          classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x

#obj 모델 



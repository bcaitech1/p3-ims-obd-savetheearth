import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import segmentation_models_pytorch as smp

import numpy as np
import timm
from pprint import pprint


class FCN8s(nn.Module):
    '''
        Backbone: VGG-16
        num_class: segmentation하고 싶은 객체의 종류        
        forward output
            - output  : [batch_size, num_classes, height, width]
    '''
    def __init__(self, num_classes=12):
        super(FCN8s, self).__init__()
        self.num_classes = num_classes

        backbone = torchvision.models.vgg16(pretrained=True)
        self.conv1 = nn.Sequential(*(list(backbone.features[0:5]))) # 1 / 2
        self.conv2 = nn.Sequential(*(list(backbone.features[5:10]))) # 1 / 4
        self.conv3 = nn.Sequential(*(list(backbone.features[10:17]))) # 1 / 8
        self.conv4 = nn.Sequential(*(list(backbone.features[17:24]))) # 1 / 16
        self.conv5 = nn.Sequential(*(list(backbone.features[24:31]))) # 1 / 32

        self.fc6 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1, stride=1, padding=0),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d())
                                    
        self.fc7 = nn.Sequential(nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, stride=1, padding=0),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d())

        self.score_3 = nn.Conv2d(in_channels=256, out_channels=self.num_classes, kernel_size=1, stride=1, padding=0)
        self.score_4 = nn.Conv2d(in_channels=512, out_channels=self.num_classes, kernel_size=1, stride=1, padding=0)
        self.score_5 = nn.Conv2d(in_channels=4096, out_channels=self.num_classes, kernel_size=1, stride=1, padding=0)

        # input, kernel, padding, stride의 i,k,p,s
        # o' = s(i'-1) + k - 2p
        self.upscore2_1 = nn.ConvTranspose2d(in_channels=self.num_classes, out_channels=self.num_classes, kernel_size=4, stride=2, padding=1)
        self.upscore2_2 = nn.ConvTranspose2d(in_channels=self.num_classes, out_channels=self.num_classes, kernel_size=4, stride=2, padding=1)
        self.upscore8_3 = nn.ConvTranspose2d(in_channels=self.num_classes, out_channels=self.num_classes, kernel_size=16, stride=8, padding=4)

        self._initialize_weights()


    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)

        fc6_out = self.fc6(conv5_out)
        fc7_out = self.fc7(fc6_out)
        score_1 = self.score_5(fc7_out)
        score_1_up = self.upscore2_1(score_1)

        score_2 = self.score_4(conv4_out)
        skip_connection_1 = score_1_up + score_2
        score_2_up = self.upscore2_2(skip_connection_1)

        score_3 = self.score_3(conv3_out)
        skip_connection_2 = score_2_up + score_3
        output = self.upscore8_3(skip_connection_2)

        return output


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = self._get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


    def _get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        """
            Make a 2D bilinear kernel suitable for upsampling
        """
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
            (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                        dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :] = filt

        return torch.from_numpy(weight).float()


class VGG16(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(self._conv3x3_relu(3, 64),
                                      self._conv3x3_relu(64, 64),
                                      nn.MaxPool2d(3, stride=2, padding=1), # 1/2
                                      self._conv3x3_relu(64, 128),
                                      self._conv3x3_relu(128, 128),
                                      nn.MaxPool2d(3, stride=2, padding=1), # 1/4
                                      self._conv3x3_relu(128, 256),
                                      self._conv3x3_relu(256, 256),
                                      self._conv3x3_relu(256, 256),
                                      nn.MaxPool2d(3, stride=2, padding=1), # 1/8
                                      self._conv3x3_relu(256, 512),
                                      self._conv3x3_relu(512, 512),
                                      self._conv3x3_relu(512, 512),
                                      nn.MaxPool2d(3, stride=1, padding=1), # stride를 1로 해서 사이즈 유지
                                      self._conv3x3_relu(512, 512, rate=2), # dilated rate = 2
                                      self._conv3x3_relu(512, 512, rate=2),
                                      self._conv3x3_relu(512, 512, rate=2),
                                      nn.MaxPool2d(3, stride=1, padding=1), # stride를 1로 해서 사이즈 유지
                                      nn.AvgPool2d(3, stride=1, padding=1)) # stride를 1로 해서 사이즈 유지

        if pretrained:
            backbone = torchvision.models.vgg16(pretrained=True)
            weight  = backbone.state_dict()
            weight2_keys = list(self.features.state_dict().keys())
            weight2 = dict()
            for idx, key in enumerate(list(weight.keys())[:26]):
                weight2[weight2_keys[idx]] = weight[key]
            self.features.load_state_dict(weight2)


    def forward(self, x):
        output = self.features(x)
        return output


    def _conv3x3_relu(self, inplanes, planes, rate=1):
        conv3x3_relu = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=rate, dilation=rate),
                                     nn.ReLU())
        return conv3x3_relu


class Atrous_module_2(nn.Module):
    def __init__(self, inplanes, num_classes, rate):
        super(Atrous_module_2, self).__init__()
        planes = inplanes
        self.atrous = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=rate, dilation=rate),
                                    nn.ReLU(),
                                    nn.Dropout2d(),
                                    nn.Conv2d(planes, planes, kernel_size=1, stride=1),
                                    nn.ReLU(),
                                    nn.Dropout2d(),
                                    nn.Conv2d(planes, num_classes, kernel_size=1, stride=1))
        self._init_parameters()


    def forward(self, x):
        output = self.atrous(x)
        return output


    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)


class DeepLabV2(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DeepLabV2, self).__init__()
        self.backbone = VGG16(pretrained=pretrained)

        rates = [6, 12, 18, 24]
        self.aspp1 = Atrous_module_2(512 , num_classes, rate=rates[0])
        self.aspp2 = Atrous_module_2(512 , num_classes, rate=rates[1])
        self.aspp3 = Atrous_module_2(512 , num_classes, rate=rates[2])
        self.aspp4 = Atrous_module_2(512 , num_classes, rate=rates[3])
        self.global_avg_pool = ASPPPooling(512, outplanes)
        
    def forward(self, x):
        x = self.backbone(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x_sum = x1 + x2 + x3 + x4
        output = F.interpolate(x_sum, scale_factor=8, mode='bilinear')

        return output

# -------------------------------------------------------------------------------------------

class Atrous_module_3(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, padding, dilation):
        super(Atrous_module_3, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.batch_norm = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()
        
        self._init_parameters()


    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d): # init BN
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=12, inplanes=512, outplanes=256, pretrained=True):
        super(DeepLabV3, self).__init__()
        self.backbone = VGG16(pretrained=pretrained)

        rates = [1, 6, 12, 18]
        self.aspp1 = Atrous_module_3(inplanes, outplanes, kernel_size=1, padding=0, dilation=rates[0])
        self.aspp2 = Atrous_module_3(inplanes, outplanes, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.aspp3 = Atrous_module_3(inplanes, outplanes, kernel_size=3, padding=rates[2], dilation=rates[2])
        self.aspp4 = Atrous_module_3(inplanes, outplanes, kernel_size=3, padding=rates[3], dilation=rates[3])
        self.image_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)),
                                        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(outplanes),
                                        nn.ReLU())

        self.fc1 = nn.Sequential(nn.Conv2d(outplanes * 5, outplanes, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(outplanes),
                                 nn.ReLU(),
                                 nn.Dropout2d())
        self.fc2 = nn.Sequential(nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(outplanes),
                                 nn.ReLU(),
                                 nn.Conv2d(outplanes, num_classes, kernel_size=1, stride=1))
        
        self._init_parameters()

    def forward(self, x):
        x = self.backbone(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.image_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)

        return x 

    
    def _init_parameters(self):
        blocks = [self.image_pool, self.fc1, self.fc2]
        for block in blocks:
            for m in block.modules():
                if isinstance(m, nn.Conv2d): # init conv
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d): # init BN
                    nn.init.constant_(m.weight,1)
                    nn.init.constant_(m.bias,0)


# -------------------------------------------------------------------------------------------
class TorchVisionDeepLabv3_ResNet101(nn.Module):
    """
        DeepLabv3 class with custom head
        Args:
            outputchannels (int, optional): The number of output channels
            in your dataset masks. Defaults to 1.
    """
    def __init__(self, num_classes=12):
        super(TorchVisionDeepLabv3_ResNet101, self).__init__()
        self.seg_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.seg_model.classifier = DeepLabHead(2048, num_classes)

    def forward(self, x):
        x = self.seg_model(x)
        return x['out']


class TorchVisionDeepLabv3_ResNet50(nn.Module):
    """
        DeepLabv3 class with custom head
        Args:
            outputchannels (int, optional): The number of output channels
            in your dataset masks. Defaults to 1.
    """
    def __init__(self, num_classes=12):
        super(TorchVisionDeepLabv3_ResNet50, self).__init__()
        self.seg_model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.seg_model.classifier = DeepLabHead(2048, num_classes)

    def forward(self, x):
        x = self.seg_model(x)
        return x['out']

# --------------------------------------------------------------------------------------------

class SMP_DeepLabV3Plus_ResNet101(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_DeepLabV3Plus_ResNet101, self).__init__()
        self.seg_model = smp.DeepLabV3Plus(encoder_name="resnet101",
                                          encoder_weights="imagenet",
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


class SMP_DeepLabV3Plus_resnext101_32x8d(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_DeepLabV3Plus_resnext101_32x8d, self).__init__()
        self.seg_model = smp.DeepLabV3Plus(encoder_name="resnext101_32x8d",
                                          encoder_weights="ssl", # ssl: emi-supervised learning on ImageNet 
                                          in_channels=3,
                                          classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


class SMP_DeepLabV3Plus_resnext101_32x16d(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_DeepLabV3Plus_resnext101_32x16d, self).__init__()
        self.seg_model = smp.DeepLabV3Plus(encoder_name="resnext101_32x16d",
                                          encoder_weights="ssl", # ssl: emi-supervised learning on ImageNet 
                                          in_channels=3,
                                          classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


# ResNest encoders do not support dilated mode
class SMP_DeepLabV3Plus_timm_resnest101e(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_DeepLabV3Plus_timm_resnest101e, self).__init__()
        self.seg_model = smp.DeepLabV3Plus(encoder_name="timm-resnest101e",
                                          encoder_weights="imagenet",
                                          in_channels=3,
                                          classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x




class SMP_DeepLabV3Plus_efficientnet_b1(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_DeepLabV3Plus_efficientnet_b1, self).__init__()
        self.seg_model = smp.DeepLabV3Plus(encoder_name="efficientnet-b1",
                                          encoder_weights="imagenet",
                                          in_channels=3,
                                          classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


class SMP_DeepLabV3Plus_se_resnext101_32x4d(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_DeepLabV3Plus_se_resnext101_32x4d, self).__init__()
        self.seg_model = smp.DeepLabV3Plus(encoder_name="se_resnext101_32x4d",
                                          encoder_weights="imagenet",
                                          in_channels=3,
                                          classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


class SMP_DeepLabV3Plus_xception(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_DeepLabV3Plus_xception, self).__init__()
        self.seg_model = smp.DeepLabV3Plus(encoder_name="xception",
                                          encoder_weights="imagenet",
                                          in_channels=3,
                                          classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x
# ---------------------------------------------------------------

class SMP_PSPNet_resnext101_32x4d(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_PSPNet_resnext101_32x4d, self).__init__()
        self.seg_model = smp.PSPNet(encoder_name="resnext101_32x4d",
                                          encoder_weights="ssl",
                                          in_channels=3,
                                          classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x

#----------------------------------------------------------------------

class SMP_UNet_effb4(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_UNet_effb4, self).__init__()
        self.seg_model = smp.Unet(encoder_name="efficientnet-b4",
                                  encoder_weights="imagenet",
                                  in_channels=3,
                                  classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


class SMP_UNet_effb4_ns(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_UNet_effb4_ns, self).__init__()
        self.seg_model = smp.Unet(encoder_name="timm-efficientnet-b4",
                                  encoder_weights="noisy-student",
                                  in_channels=3,
                                  classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


class SMP_UNet_resnext101_32x4d(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_UNet_resnext101_32x4d, self).__init__()
        self.seg_model = smp.Unet(encoder_name="resnext101_32x4d",
                                  encoder_weights="ssl",
                                  in_channels=3,
                                  classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x

# ------------------------------------------------------

class SMP_Linknet_se_resnext50_32x4d(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_Linknet_se_resnext50_32x4d, self).__init__()
        self.seg_model = smp.Unet(encoder_name="se_resnext50_32x4d",
                                  encoder_weights="imagenet",
                                  in_channels=3,
                                  classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x

# ------------------------------------------------------

class SMP_FPN_effb0(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_FPN_effb0, self).__init__()
        self.seg_model = smp.FPN(encoder_name="efficientnet-b0",
                                  encoder_weights="imagenet",
                                  in_channels=3,
                                  classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


class SMP_FPN_effb1(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_FPN_effb1, self).__init__()
        self.seg_model = smp.FPN(encoder_name="efficientnet-b1",
                                  encoder_weights="imagenet",
                                  in_channels=3,
                                  classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


class SMP_FPN_effb2(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_FPN_effb2, self).__init__()
        self.seg_model = smp.FPN(encoder_name="efficientnet-b2",
                                  encoder_weights="imagenet",
                                  in_channels=3,
                                  classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


class SMP_FPN_effb3(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_FPN_effb3, self).__init__()
        self.seg_model = smp.FPN(encoder_name="efficientnet-b3",
                                  encoder_weights="imagenet",
                                  in_channels=3,
                                  classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


class SMP_FPN_effb3_ns(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_FPN_effb3_ns, self).__init__()
        self.seg_model = smp.FPN(encoder_name="timm-efficientnet-b3",
                                  encoder_weights="noisy-student",
                                  in_channels=3,
                                  classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


class SMP_FPN_effb4(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_FPN_effb4, self).__init__()
        self.seg_model = smp.FPN(encoder_name="efficientnet-b4",
                                  encoder_weights="imagenet",
                                  in_channels=3,
                                  classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


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


class SMP_FPN_effb5_ns(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_FPN_effb5_ns, self).__init__()
        self.seg_model = smp.FPN(encoder_name="timm-efficientnet-b5",
                                  encoder_weights="noisy-student",
                                  in_channels=3,
                                  classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


class SMP_FPN_effb5(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_FPN_effb5, self).__init__()
        self.seg_model = smp.FPN(encoder_name="efficientnet-b5",
                                  encoder_weights="imagenet",
                                  in_channels=3,
                                  classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x

    
class SMP_FPN_effb6(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_FPN_effb6, self).__init__()
        self.seg_model = smp.FPN(encoder_name="efficientnet-b6",
                                  encoder_weights="imagenet",
                                  in_channels=3,
                                  classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


class SMP_FPN_resnext101_32x4d(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_FPN_resnext101_32x4d, self).__init__()
        self.seg_model = smp.FPN(encoder_name="resnext101_32x4d",
                                          encoder_weights="ssl",
                                          in_channels=3,
                                          classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x


class SMP_FPN_resnext101_32x8d(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_FPN_resnext101_32x8d, self).__init__()
        self.seg_model = smp.FPN(encoder_name="resnext101_32x8d",
                                          encoder_weights="ssl",
                                          in_channels=3,
                                          classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x
        

class SMP_FPN_resnet101(nn.Module):
    def __init__(self, num_classes=12):
        super(SMP_FPN_resnet101, self).__init__()
        self.seg_model = smp.FPN(encoder_name="resnet101",
                                          encoder_weights="imagenet",
                                          in_channels=3,
                                          classes=12)

    def forward(self, x):
        x = self.seg_model(x)
        return x
        
# --------------------------------------------------------------

# for checking forward progress
if __name__ == "__main__":
    # backbone = torchvision.models.vgg16(pretrained=True)
    # print(backbone)
    # x = torch.randn(2, 3, 512, 512)
    # model = FCN8s()
    # print(model)
    # output = model(x)
    # print(output.shape)

    # model = DeepLabV2(num_classes=12)
    # x = torch.randn(2, 3, 512, 512)
    # output = model(x)
    # print(output.shape)

    # model = DeepLabV3(num_classes=12)
    # x = torch.randn(2, 3, 512, 512)
    # output = model(x)
    # print(output.shape)

    model = SMP_UNet_resnext101_32x4d(num_classes=12)
    x = torch.randn(2, 3, 512, 512)
    print(model)
    output = model(x)
    print(output.shape)
    pass

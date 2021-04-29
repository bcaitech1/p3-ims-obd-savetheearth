import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import timm
from pprint import pprint
# from efficientnet_pytorch import EfficientNet
from segmentation_models_pytorch.unet import Unet
# from efficientunet import *

class EffUNet(nn.Module):
    def __init__(self, num_classes):
        super(EffUNet,self).__init__()
        self.unet = Unet('efficientnet-b5', encoder_weights="imagenet", classes=num_classes, activation=None)
        # self.unet = get_efficientunet_b5(out_channels=12, pretrained=True)

    def forward(self, x):
        score = self.unet(x)

        return score

class Deconvnet_vgg(nn.Module):
    def __init__(self, num_classes):
        super(Deconvnet_vgg,self).__init__()
        backbone = torchvision.models.vgg16(pretrained=True)
        # self.pretrained_model = vgg16(pretrained = True)
        # features, classifiers = list(self.pretrained_model.features.children()), list(self.pretrained_model.classifier.children())

        def DCB(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, 
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride, 
                                            padding=padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True)) 

        self.features_map1 = nn.Sequential(*backbone.features[0:4]) #conv1(maxpooling 전까지)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True,return_indices=True)

        self.features_map2 = nn.Sequential(*backbone.features[5:9]) #conv2(maxpooling 전까지)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True,return_indices=True)

        self.features_map3 = nn.Sequential(*backbone.features[10:16]) #conv3(maxpooling 전까지)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True,return_indices=True)

        self.features_map4 = nn.Sequential(*backbone.features[17:23]) #conv4(maxpooling 전까지)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True,return_indices=True)

        self.features_map5 = nn.Sequential(*backbone.features[24:30]) #conv5(maxpooling 전까지)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True,return_indices=True)
        
        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7, 1, 0)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(0.5)

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1, 1, 0)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(0.5)

        # fc6-deconv
        self.fc6_deconv = DCB(4096, 512, 7, 1, 0)

        #unpool5 #14*14
        self.unpool5 = nn.MaxUnpool2d(2,stride = 2)
        self.deconv5_1 = DCB(512, 512, 3, 1, 1)
        self.deconv5_2 = DCB(512, 512, 3, 1, 1)
        self.deconv5_3 = DCB(512, 512, 3, 1, 1)

        #unpool4 #28*28
        self.unpool4 = nn.MaxUnpool2d(2,stride = 2)
        self.deconv4_1 = DCB(512, 512, 3, 1, 1)
        self.deconv4_2 = DCB(512, 512, 3, 1, 1)
        self.deconv4_3 = DCB(512, 256, 3, 1, 1)
       
        #unpool4 #56*56
        self.unpool3 = nn.MaxUnpool2d(2,stride = 2)
        self.deconv3_1 = DCB(256, 256, 3, 1, 1)
        self.deconv3_2 = DCB(256, 256, 3, 1, 1)
        self.deconv3_3 = DCB(256, 128, 3, 1, 1)

        #unpool4 #112*112
        self.unpool2 = nn.MaxUnpool2d(2,stride = 2)
        self.deconv2_1 = DCB(128, 128, 3, 1, 1)
        self.deconv2_2 = DCB(128, 64, 3, 1, 1)

        #unpool4 #224*224
        self.unpool1 = nn.MaxUnpool2d(2,stride = 2)
        self.deconv1_1 = DCB(64, 64, 3, 1, 1)
        self.deconv1_2 = DCB(64, 64, 3, 1, 1)

        #Score
        self.score_fr = nn.Conv2d(64, num_classes, 1, 1, 0, 1)
        
    def forward(self, x):
        max1 = h = self.features_map1(x)
        h, pool1_indices = self.pool1(h)

        max2 = h = self.features_map2(h)
        h, pool2_indices = self.pool2(h)

        max3 = h = self.features_map3(h)
        h, pool3_indices = self.pool3(h)

        max4 = h = self.features_map4(h)
        h, pool4_indices = self.pool4(h)

        max5 = h = self.features_map5(h)
        h, pool5_indices = self.pool5(h)

        #fc6
        h = self.fc6(h)
        h = self.relu6(h)
        h = self.drop6(h)

        #fc7
        h = self.fc7(h)
        h = self.relu7(h)
        h = self.drop7(h)     

        #fc6-deconv
        h = self.fc6_deconv(h)

        #deconv5
        h = self.unpool5(h, pool5_indices)
        h = self.deconv5_1(h)
        h = self.deconv5_2(h)
        h = self.deconv5_3(h)

        #deconv4
        h = self.unpool4(h, pool4_indices)
        h = self.deconv4_1(h)
        h = self.deconv4_2(h)
        h = self.deconv4_3(h)

        #deconv3
        h = self.unpool3(h, pool3_indices)
        h = self.deconv3_1(h)
        h = self.deconv3_2(h)
        h = self.deconv3_3(h)

        #deconv2
        h = self.unpool2(h, pool2_indices)
        h = self.deconv2_1(h)
        h = self.deconv2_2(h)

        #deconv1
        h = self.unpool1(h, pool1_indices)
        h = self.deconv1_1(h)
        h = self.deconv1_2(h)

        score = self.score_fr(h)

        return score

        
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


# for checking forward progress
if __name__ == "__main__":
    # backbone = torchvision.models.vgg16(pretrained=True)
    # print(backbone)
    # x = torch.randn(2, 3, 512, 512)
    # model = FCN8s()
    # print(model)
    # output = model(x)
    # print(output.shape)
    pass
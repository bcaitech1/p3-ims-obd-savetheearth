import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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




# for checking forward progress
if __name__ == "__main__":
    # backbone = torchvision.models.vgg16(pretrained=True)
    # print(backbone)
    # x = torch.randn(2, 3, 512, 512)
    model = FCN8s()
    print(model)
    # output = model(x)
    # print(output.shape)
    pass
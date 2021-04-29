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


class SegNet(nn.Module):
    def __init__(self, num_classes=12, init_weights=True, pretrained=True):
        super(SegNet, self).__init__()
        self.pretrained = pretrained

        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, 
                                 out_channels=out_channels,
                                 kernel_size=kernel_size, 
                                 stride=stride, 
                                 padding=padding)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)
            return cbr
        
        # conv1 
        self.cbr1_1 = CBR(3, 64, 3, 1, 1)
        self.cbr1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True) 

        # conv2 
        self.cbr2_1 = CBR(64, 128, 3, 1, 1)
        self.cbr2_2 = CBR(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True) 

        # conv3
        self.cbr3_1 = CBR(128, 256, 3, 1, 1)
        self.cbr3_2 = CBR(256, 256, 3, 1, 1)
        self.cbr3_3 = CBR(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True) 

        # conv4
        self.cbr4_1 = CBR(256, 512, 3, 1, 1)
        self.cbr4_2 = CBR(512, 512, 3, 1, 1)
        self.cbr4_3 = CBR(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True) 

        # conv5
        self.cbr5_1 = CBR(512, 512, 3, 1, 1)
        self.cbr5_2 = CBR(512, 512, 3, 1, 1)
        self.cbr5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True) 
        
        # deconv5
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr5_3 = CBR(512, 512, 3, 1, 1)
        self.dcbr5_2 = CBR(512, 512, 3, 1, 1)
        self.dcbr5_1 = CBR(512, 512, 3, 1, 1)

        # deconv4 
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr4_3 = CBR(512, 512, 3, 1, 1)
        self.dcbr4_2 = CBR(512, 512, 3, 1, 1)
        self.dcbr4_1 = CBR(512, 256, 3, 1, 1)

        # deconv3
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr3_3 = CBR(256, 256, 3, 1, 1)
        self.dcbr3_2 = CBR(256, 256, 3, 1, 1)
        self.dcbr3_1 = CBR(256, 128, 3, 1, 1)

        # deconv2
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr2_2 = CBR(128, 128, 3, 1, 1)
        self.dcbr2_1 = CBR(128, 64, 3, 1, 1)

        # deconv1
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1_1 = CBR(64, 64, 3, 1, 1)
        # Score
        # self.score_fr = nn.Conv2d(64, num_classes, kernel_size = 1)
        self.score_fr = nn.Conv2d(64, num_classes, kernel_size = 3, padding=1)
        
        if init_weights:
            self._initialize_weights()
        
        if self.pretrained:
            backbone = torchvision.models.vgg16(pretrained=self.pretrained)
            weight = backbone.state_dict()
            cnt = 0
            wnb = ['weight', 'bias']
            weight2 = self.state_dict()
            weight_next = dict()
            weight2_wnb = list()
            for i, key in enumerate(weight2.keys()):
                conds = key.split('.')
                if conds[-1] in wnb and conds[-2] == '0':
                    weight2_wnb.append(key)
                    cnt += 1
                if cnt == 26:
                    break
            for idx, key in enumerate(list(weight.keys())[:26]):
                weight_next[weight2_wnb[idx]] = weight[key]
            self.load_state_dict(weight_next, strict=False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        h = self.cbr1_1(x)
        h = self.cbr1_2(h)
        dim1 = h.size()
        h, pool1_indices = self.pool1(h)
        
        h = self.cbr2_1(h)
        h = self.cbr2_2(h)
        dim2 = h.size()
        h, pool2_indices = self.pool2(h)
        
        h = self.cbr3_1(h)
        h = self.cbr3_2(h)
        h = self.cbr3_3(h)
        dim3 = h.size()
        h, pool3_indices = self.pool3(h)
        
        h = self.cbr4_1(h)
        h = self.cbr4_2(h)
        h = self.cbr4_3(h)
        dim4 = h.size()
        h, pool4_indices = self.pool4(h)
        
        h = self.cbr5_1(h)
        h = self.cbr5_2(h)
        h = self.cbr5_3(h)
        dim5 = h.size()
        h, pool5_indices = self.pool5(h)
        
        h = self.unpool5(h, pool5_indices, output_size = dim5)
        h = self.dcbr5_3(h)
        h = self.dcbr5_2(h)
        h = self.dcbr5_1(h)
        
        h = self.unpool4(h, pool4_indices, output_size = dim4)
        h = self.dcbr4_3(h)
        h = self.dcbr4_2(h)
        h = self.dcbr4_1(h)
        
        h = self.unpool3(h, pool3_indices, output_size = dim3)
        h = self.dcbr3_3(h)
        h = self.dcbr3_2(h)
        h = self.dcbr3_1(h)
        
        h = self.unpool2(h, pool2_indices, output_size = dim2)
        h = self.dcbr2_2(h)
        h = self.dcbr2_1(h)
        
        h = self.unpool1(h, pool1_indices, output_size = dim1)
        h = self.deconv1_1(h)
        out = self.score_fr(h) 
        
        return out

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
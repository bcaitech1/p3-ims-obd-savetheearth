from django.shortcuts import render,redirect
from django.http import HttpResponse
from .models import Input
from django.conf import settings

import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch.transforms import ToTensorV2

import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import os

from .visualize import log_images
from .detect_model.detection_result import detection

MODEL_DIR='./p3_app/static/models'
SEG_DIR='../media/Seg'
DET_DIR='../media/Det'

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

def upload(request):
    return render(request,'index.html')

def seg_result(img):
    seg_model = SMP_FPN_effb4_ns()
    seg_model.load_state_dict(torch.load(os.path.join(MODEL_DIR,'FPN_effb4_ns_Fold0.pt')))
    seg_model.eval()

    pil_image = Image.open(img)
    img=pil_image.transpose(Image.ROTATE_270)
    image = np.array(img)

    transform = Compose([
                        Resize(512,512),
                        Normalize(
                            mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
                        ToTensorV2()])
    image = transform(image=image)['image'].unsqueeze(0)
    output = seg_model(image)

    return output

def upload_create(request):
    form=Input()
    try:
        form.image=request.FILES['image']
    except: #이미지가 없을 때
        pass
    form.save() #db에 이미지 저장
    
    #segmentation
    seg_res_logits = seg_result(request.FILES['image'])  
    probs = F.softmax(seg_res_logits, dim=1)
    probs = probs.data.cpu().numpy()
    preds = np.argmax(probs, axis=1)
    seg_res = log_images(preds, form.image) #PIL image리턴
    Seg_Path = os.path.join(settings.MEDIA_ROOT,'Seg',request.FILES['image'].name)
    seg_res.save(Seg_Path)
    
    #detection
    det_res=detection(request.FILES['image'])
    Det_Path=os.path.join(settings.MEDIA_ROOT,'Det',request.FILES['image'].name)
    det_res.save(Det_Path)

    return render(request,'result.html',{'seg_res_path':'../media/Seg/'+request.FILES['image'].name,
                                         'det_res_path':'../media/Det/'+request.FILES['image'].name})


# pip install mmcv-full==1.3.7 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html

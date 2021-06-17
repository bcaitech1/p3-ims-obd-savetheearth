from django.shortcuts import render,redirect
from django.http import HttpResponse
from .models import Input
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from albumentations import Compose, Normalize
from albumentations.pytorch.transforms import ToTensorV2
from .visualize import log_images

def upload(request):
    return render(request,'index.html')

def seg_result(img):
    seg_model = torch.load('./p3_app/static/p3_models/seg_model.pt')
    seg_model.eval()

    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = Compose([Normalize(
                        mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
                        ToTensorV2()])
    image = transform(image=image)['image'].unsqueeze(0)
    output = seg_model(image)

    return output

def obj_result(img):
    obj_model = torch.load('./p3_app/static/p3_models/detection_model.pt')
    obj_model.eval()

    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = Compose([Normalize(
                                mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
                                    ToTensorV2()])
    image = transform(image=image)['image'].unsqueeze(0)
    img_dict = {}
    img_dict['img_metas'] = [0]
    img_dict['img'] = image
    output = obj_model(img_dict)

    return output


def upload_create(request):
    form=Input()
    try:
        form.image=request.FILES['image']
    except: #이미지가 없을 때
        pass
    form.save() #db에 이미지 저장

    seg_res_logits = seg_result(form.image)
    viz_seg_res = log_images(seg_res_logits, form.image)

    obj_res_output = obj_result(form.image)
    viz_obj_res = 99999999999 ###################### 수정

    return render(request,'result.html',{'seg_res_fig':viz_seg_res, 'obj_res_fig':viz_obj_res})
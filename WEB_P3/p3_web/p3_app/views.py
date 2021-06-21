import re
from django.shortcuts import render,redirect
from django.http import HttpResponse,JsonResponse
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
import json
import pickle
import random

from .visualize import log_images
from .detect_model.detection_result import detection
from pprint import pprint

MODEL_DIR='./p3_app/static/models'
SEG_DIR='media/Seg'
DET_DIR='media/Det'
ORIGIN_DIR='media/DATA'
RESULT_DIR='media/RESULT'

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
    # 이미지 저장할 폴더 만들기
    os.makedirs(SEG_DIR, exist_ok=True)
    os.makedirs(DET_DIR, exist_ok=True)
    os.makedirs(ORIGIN_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    colors=['#c8c8c8','#4d7bb3','#65b4ba','#35bd50','#8530b3','#eddd28',
            '#e3a127','#e207e6','#ff008c','#63b9ff','#6f1cff','#c6ff1c']
    classes=["Background","UNKNOWN","General trash","Paper","Paper pack",
    	    "Metal","Glass","Plastic","Styrofoam","Plastic bag","Battery","Clothing"]
    threshold=[0.5,0.6,0.75,0.9]
    list1,list2,list3,list4={},{},{},{}
    for i,(color,_class) in enumerate(zip(colors,classes)):
        if i < 3:
            list1[color]=_class
        elif i<6:
            list2[color]=_class
        elif i<9:
            list3[color]=_class
        else:
            list4[color]=_class

    context={'list1':list1,'list2':list2,'list3':list3,'list4':list4,
            'classes':classes[1:],'threshold':threshold}
    return render(request,'index.html',context)

def seg_result(img):
    seg_model = SMP_FPN_effb4_ns()
    seg_model.load_state_dict(torch.load(os.path.join(MODEL_DIR,'FPN_effb4_ns_Fold0.pt')))
    seg_model.eval()

    transform = Compose([
                        Resize(512,512),
                        Normalize(
                            mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
                        ToTensorV2()])
    img = transform(image=img)['image'].unsqueeze(0)
    output = seg_model(img)

    return output

def upload_create(request):
    if request.is_ajax():
        print('ajax 통신!!')
        form=Input()
        try:
            form.image=request.FILES['image']
        except: #이미지가 없을 때
            pass
        form.save() #db에 이미지 저장

        # input image로 변환
        img = Image.open(request.FILES['image']).convert('RGB')
        #img = img.transpose(Image.ROTATE_270)
        img = np.array(img)
        # 원본 데이터 저장하기
        idx=request.FILES['image'].name.find('.')
        pickle_name=request.FILES['image'].name[:idx]+'.pkl'
        origin_data_path=os.path.join(ORIGIN_DIR,pickle_name)
        with open(origin_data_path,'wb') as f:
            pickle.dump(img,f)


        #segmentation
        seg_res_logits = seg_result(img)  
        probs = F.softmax(seg_res_logits, dim=1)
        probs = probs.data.cpu().numpy()
        preds = np.argmax(probs, axis=1)
        seg_res = log_images(preds, img) #PIL image리턴
        Seg_Path = os.path.join(settings.MEDIA_ROOT,'Seg',request.FILES['image'].name)
        seg_res.save(Seg_Path)
        
        #detection
        det_res , raw_res=detection(img)
        Det_Path=os.path.join(settings.MEDIA_ROOT,'Det',request.FILES['image'].name)
        result_data_path=os.path.join(RESULT_DIR,pickle_name)
        with open(result_data_path,'wb') as f:
            pickle.dump(raw_res,f)
        det_res.save(Det_Path)

        context = {'seg_res_path':'../media/Seg/'+request.FILES['image'].name,
                    'det_res_path':'../media/Det/'+request.FILES['image'].name,
                    'origin_data_path':origin_data_path,
                    'result_data_path':result_data_path}
        return HttpResponse(json.dumps(context), content_type='application/json') 


def detect_by_class(request):
    if request.is_ajax():
        print(request.POST,'post!!!!!!')
        print(request.FILES,'files!!!')
        thr=float(request.POST['thr'])
        cls=request.POST['cls']
        origin_data=request.POST['origin_data']
        result_data=request.POST['result_data']
        print(thr,cls)
        print(origin_data)
        print(result_data)
        with open( origin_data, "rb" ) as f:
            origin = pickle.load(f)
        with open( result_data, "rb" ) as f:
            result = pickle.load(f)

        det_res , _ =detection(origin,result,thr,cls)
        r=str(random.randint(0,10000))
        tmp_path=os.path.join(DET_DIR,r+'.jpg')
        det_res.save(tmp_path)
        context={'det_res_path':'../'+tmp_path}
        return HttpResponse(json.dumps(context), content_type='application/json')
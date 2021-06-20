from mmdet.apis import inference_detector, init_detector, show_result_pyplot

from PIL import Image
from albumentations import Resize
import matplotlib.pyplot as plt
import numpy as np
import os

def detection(img):
    colors =[
        (129, 236, 236),
        (2, 132, 227),
        (232, 67, 147),
        (253, 234, 255),
        (0, 184, 148),
        (85, 239, 196),
        (48, 51, 107),
        (253, 159, 26),
        (253, 204, 204),
        (179, 57, 57),
        (248, 243, 212)
    ]
    
    # Choose to use a config and initialize the detector
    print(os.path.realpath(__file__))
    config = '/opt/ml/p3-ims-obd-savetheearth/WEB_P3/p3_web/p3_app/detect_model/configs/My_config/vfnet_x101_64x4d_fpn_ver2.py'
    # Setup a checkpoint file to load
    checkpoint = '/opt/ml/p3-ims-obd-savetheearth/WEB_P3/p3_web/p3_app/detect_model/vfnet_x101_64x4d_fpn.pth'
    # initialize the detector
    model = init_detector(config, checkpoint, device='cuda:0')
    
    Re=Resize(512,512)
    img=Re(image=img)['image']
    
    result = inference_detector(model, img)
    show_result=model.show_result(img,result,score_thr=0.6,bbox_color=colors,text_color='white',show=False,thickness=3)
    save_img=Image.fromarray(show_result)
    return save_img
    
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from albumentations import Resize

def log_images(masks,image):
    colors =[
        [200, 200, 200],
        [129, 236, 236],
        [2, 132, 227],
        [232, 67, 147],
        [255, 234, 267],
        [0, 184, 148],
        [85, 239, 196],
        [48, 51, 107],
        [255, 159, 26],
        [255, 204, 204],
        [179, 57, 57],
        [248, 243, 212],
    ]
    colors = np.array(colors).astype('uint8')
    Re=Resize(512,512)
    img=Re(image=image)['image']
    answer = ((0.4 * img) + (0.6 * colors[masks[0]])).astype('uint8')
    result_img = Image.fromarray(answer)

    return result_img
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from albumentations import Resize

def log_images(masks,image):
    colors =[
            [200, 200, 200],
            [77, 123, 179],
            [101, 180, 186],
            [53, 189, 80],
            [133, 48, 179],
            [237, 221, 40],
            [227, 161, 39],
            [226, 7, 230],
            [255, 0, 140],
            [99, 185, 255],
            [111, 28, 255],
            [198, 255, 28]
    ]
    colors = np.array(colors).astype('uint8')
    Re=Resize(512,512)
    img=Re(image=image)['image']
    answer = ((0.4 * img) + (0.6 * colors[masks[0]])).astype('uint8')
    result_img = Image.fromarray(answer)

    return result_img
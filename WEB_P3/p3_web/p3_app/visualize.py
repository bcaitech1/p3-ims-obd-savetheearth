import numpy as np
import os
import matplotlib.pyplot as plt

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

    fig, axes = plt.subplots(2, 2, figsize=(3*2, 3*2))
    for i in range(2):
      image = np.array(image)

      answer = ((0.4 * image) + (0.6 * colors[masks[i]])).astype('uint8')
      # answer = masks[i]

      axes[0,i].imshow(answer)
      axes[0,i].set_title(np.unique(masks[i]))

    return fig
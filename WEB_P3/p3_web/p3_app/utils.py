import numpy as np

#seg 결과 시각화
def log_images(masks, preds, img_info):
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

    fig, axes = plt.subplots(CFG.valid_batch_size, 2, figsize=(3*2, 3*CFG.valid_batch_size))
    for i in range(CFG.valid_batch_size):
        image = np.array(Image.open(os.path.join(CFG.BASE_DATA_PATH, img_info[i]["file_name"])))

        answer = ((0.4 * image) + (0.6 * colors[masks[i]])).astype('uint8')
        prediction = ((0.4 * image) + (0.6 * colors[preds[i]])).astype('uint8')

        axes[i,0].imshow(answer)
        axes[i,0].set_title(np.unique(masks[i]))

        axes[i,1].imshow(prediction)
        axes[i,1].set_title(np.unique(preds[i]))

    fig.tight_layout()
    return fig

from albumentations import (Compose, Resize, Normalize, Rotate, CenterCrop, RandomBrightnessContrast, RandomContrast, GaussNoise, HorizontalFlip, pytorch)
from albumentations.pytorch import ToTensorV2

class BaseTrainAugmentation:
    def __init__(self):
        self.transformer = Compose([
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        return self.transformer(image=image, mask=mask)


class BaseTestAugmentation:
    def __init__(self):
        self.transformer = Compose([
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(),
        ])

    def __call__(self, image):
        return self.transformer(image=image)


class CenterCropBaseAugmentation:
    def __init__(self, resize_height, resize_width):
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.transformer = Compose([
            CenterCrop(height = self.resize_height, width = self.resize_width),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            albumentations.pytorch.transforms.ToTensor(),
        ])

    def __call__(self, image):
        return self.transformer(image=image)


class ResizeVariousAugmentation:
    def __init__(self, resize_height, resize_width):
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.transformer = Compose([
            Resize(height = self.resize_height, width = self.resize_width),
            RandomContrast(limit=[0.5,0.51],always_apply=True),
            HorizontalFlip(p=0.5),
            Rotate(limit=5, p=0.5),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            albumentations.pytorch.transforms.ToTensor(),
        ])

    def __call__(self, image):
        return self.transformer(image=image)



class CenterCropVariousAugmentation:
    def __init__(self, resize_height, resize_width):
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.transformer = Compose([
            CenterCrop(height = self.resize_height, width = self.resize_width, always_apply=True),
            RandomBrightnessContrast(p=0.5),
            HorizontalFlip(p=0.5),
            Rotate(limit=3, p=0.5),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            albumentations.pytorch.transforms.ToTensor(),
        ])

    def __call__(self, image):
        return self.transformer(image=image)
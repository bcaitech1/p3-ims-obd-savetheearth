from albumentations import (Compose, Resize, Normalize, ShiftScaleRotate, Rotate, GridDistortion, CenterCrop, RandomResizedCrop, CLAHE, RandomBrightnessContrast, ElasticTransform, RandomContrast, GaussNoise, HorizontalFlip, pytorch, Cutout, VerticalFlip, OneOf, CropNonEmptyMaskIfExists)
from albumentations.pytorch import ToTensorV2

class BaseTrainAugmentation:
    def __init__(self):
        self.transformer = Compose([
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        return self.transformer(image=image, mask=mask)


class Aug1TrainAugmentation:
    def __init__(self):
        self.transformer = Compose([
            OneOf([
                VerticalFlip(),
                HorizontalFlip(),
            ], p=0.5),
            
            Cutout(num_holes=8, max_h_size=20, max_w_size=20, p=0.5),
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        return self.transformer(image=image, mask=mask)


class Aug2TrainAugmentation:
    def __init__(self):
        self.transformer = Compose([
            OneOf([
                VerticalFlip(),
                HorizontalFlip(),
            ], p=0.5),
            ElasticTransform(always_apply=False, p=0.3, alpha=1.68, sigma=48.32, alpha_affine=44.97, interpolation=0, border_mode=2, value=(0, 0, 0), mask_value=None, approximate=False),
            Cutout(num_holes=8, max_h_size=20, max_w_size=20, p=0.5),
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        return self.transformer(image=image, mask=mask)


class Aug3TrainAugmentation:
    def __init__(self):
        self.transformer = Compose([
            OneOf([
                VerticalFlip(),
                HorizontalFlip(),
            ], p=0.5),
            GridDistortion(always_apply=False, p=0.5, num_steps=5, distort_limit=(-0.46, 0.40), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
            Cutout(num_holes=8, max_h_size=20, max_w_size=20, p=0.5),
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        return self.transformer(image=image, mask=mask)


class Aug4TrainAugmentation:
    def __init__(self):
        self.transformer = Compose([
            OneOf([
                VerticalFlip(),
                HorizontalFlip(),
            ], p=0.5),
            OneOf([
                GridDistortion(always_apply=False, p=0.5, num_steps=5, distort_limit=(-0.46, 0.40), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
                ElasticTransform(always_apply=False, p=0.5, alpha=1.68, sigma=48.32, alpha_affine=44.97, interpolation=0, border_mode=2, value=(0, 0, 0), mask_value=None, approximate=False),
            ], p=0.5),
            CLAHE(clip_limit=(1, 8), tile_grid_size=(10, 10),p=0.3),
            Cutout(num_holes=8, max_h_size=20, max_w_size=20, p=0.5),
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        return self.transformer(image=image, mask=mask)


class Aug5TrainAugmentation:
    def __init__(self):
        self.transformer = Compose([
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(always_apply=False, p=0.5, shift_limit=(-0.06, 0.06), scale_limit=(-0.10, 0.10), rotate_limit=(-15, 15), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
            GridDistortion(always_apply=False, p=0.5, num_steps=5, distort_limit=(-0.46, 0.40), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
            Cutout(num_holes=8, max_h_size=20, max_w_size=20, p=0.5),
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        return self.transformer(image=image, mask=mask)


class AugLastTrainAugmentation:
    def __init__(self):
        self.transformer = Compose([
            HorizontalFlip(p=0.5),
            CLAHE(clip_limit=(1, 8), tile_grid_size=(10, 10), p=0.3),
            OneOf([
                GridDistortion(num_steps=5, distort_limit=(-0.46, 0.40)),
                ElasticTransform(alpha=1.68, sigma=48.32, alpha_affine=44.97),
            ], p=0.3),
            RandomResizedCrop(p=0.3, height=512, width=512, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
            ShiftScaleRotate(p=0.3, shift_limit=(-0.06, 0.06), scale_limit=(-0.10, 0.10), rotate_limit=(-20, 20)),
            Normalize(mean=[0.46009655, 0.43957878, 0.41827092], std=[0.2108204, 0.20766491, 0.21656131], max_pixel_value=255.0, p = 1.0),
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        return self.transformer(image=image, mask=mask)


class FinalTrainAugmentation:
    def __init__(self):
        self.transformer = Compose([
            HorizontalFlip(p=0.5),
            OneOf([
                GridDistortion(num_steps=5, distort_limit=(-0.46, 0.40), value=(0, 0, 0)),
                ElasticTransform(alpha=1.68, sigma=48.32, alpha_affine=44.97,value=(0, 0, 0)),
                RandomResizedCrop(height=512, width=512, scale=(0.08, 1.0), ratio=(0.75, 1.33))
            ], p=.3),
            ShiftScaleRotate(shift_limit=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-90, 90),p=0.3),
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
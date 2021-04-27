import torch
from torch.utils.data import Dataset

from pycocotools.coco import COCO
import numpy as np
import cv2
import os



class RecycleDataset(Dataset):
    """
        Description: COCO format을 기반으로하는 Pytorch Dataset 
    """
    
    def __init__(self, data_path, coco_path, mode = 'train', transform = None):
        super().__init__()
        self.data_path = data_path
        self.coco = COCO(coco_path)
        self.mode = mode
        self.transform = transform

        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds = index) # Get img ids which is related to index
        image_infos = self.coco.loadImgs(image_id)[0] # Load anns list with the specified ids.
        
        # cv2 를 활용하여 image 불러오기
        image = cv2.imread(os.path.join(self.data_path, image_infos['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds = image_infos['id']) # Get ann ids which is related to image_id
            anns = self.coco.loadAnns(ann_ids) # Load anns list with the specified ids.

            # mask : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            mask = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                pixel_value = anns[i]['category_id'] + 1
                mask = np.maximum(self.coco.annToMask(anns[i]) * pixel_value, mask) # Convert annotation to binary mask
            mask = mask.astype(np.float32)

            # transform -> albumentations 라이브러리 활용
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            
            return image, mask, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            
            return image, image_infos
    
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())
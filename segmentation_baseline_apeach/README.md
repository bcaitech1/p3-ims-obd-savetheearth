# p3-ims-obd-savetheearth
# Trash image semantic segmentation

- 대회 기간 : 2021.05.10 ~ 2021.05.20
- 팀 : 지구를 지켜조 (강민용, 김채현, 노원재, 이정현, 황정현)
- 내용 : **재활용 품목 분류를 위한 Semantic Segmentation**

## ❗ 웹 데모

- 개발중

## ❗ 개요 

### 주제

재활용 품목 분류를 위한 Semantic Segmentation

### 배경

환경 부담을 조금이나마 줄일 수 있는 방법의 하나로 '분리수거'가 있습니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립, 소각되기 때문입니다. 하지만 '이 쓰레기가 어디에 속하는지', '어떤 것들을 분리해서 버리는 것이 맞는지' 등 정확한 분리수거 방법을 알기 어렵다는 문제점이 있습니다. 따라서, 쓰레기가 찍힌 사진에서 쓰레기를 Segmentation 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다.

### 평가

- 평가 metric : **mIoU**

  모든 이미지에서 계산된 mIoU를 평균내어 반영합니다. 


## ❗  데이터 

### 데이터셋의 간략한 통계

------

- 전체 이미지 개수 : 4109장

- 12 class : Background, UNKNOWN, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
  - 참고 : train_all.json/train.json/val.json에는 background에 대한 annotation이 존재하지 않으므로 background (0) class 추가 (baseline 참고)
- 이미지 크기 : (512, 512)

### **annotation file**

------

annotation file은 [coco format](https://cocodataset.org/#home) 으로 이루어져 있습니다.

[coco format](https://cocodataset.org/#home)은 크게 2가지 (images, annotations)의 정보를 가지고 있습니다.

- images:
  - id: 파일 안에서 image 고유 id, ex) 1
  - height: 512
  - width: 512
  - file*name: ex) batch*01_vt/002.jpg
- annotations: 
  - id: 파일 안에 annotation 고유 id, ex) 1
  - segmentation: masking 되어 있는 고유의 좌표
  - bbox: 객체가 존재하는 박스의 좌표 (x*min, y*min, w, h)
  - area: 객체가 존재하는 영역의 크기
  - category_id: 객체가 해당하는 class의 id
  - image_id: annotation이 표시된 이미지 고유 id



## ❗  실험 내용

### Loss

DeepLab V3 모델로 실험을 진행. Focal Loss, CrossEntropyWithL1, DiceLoss, DiceTopKLoss, DiceFocalLoss, DiceBCELoss 등 다양한 실험을 진행한 결과 기본적인 Cross Entropy가 가장 좋은 성능을 보이고 있어서 최종적으로 Cross Entropy를 사용하기로 했습니다.



### Augmentation

먼저 실험을 통해서 Augmentation을 찾기 위해 후보군을 정해서 실험을 진행했습니다.

- Horizontal Flip, ChannelShuffle, CLAHE ,Vertical Flip, RandomBrightnessContrast ,Cutout, ElasticTransform, GridDistortion, ShiftScaleRotate , RandomResizedCrop을 실험해보고 LB score에 긍정적인 영향을 주는 요인만 조합하여 확률적으로 사용했습니다.

- 최종 Augmentation

  - HorizontalFlip

  - GridDistortion

  - ElasticTransform

  - RandomResizedCrop

  - ShiftScaleRotate

  - Normalize

    

### Model

- **Deeplab V3 Plus ( backbone: resnext101_32x4d )**

  \- Learning rate : Encoder 0.001 * 0.1 / Decoder + Segmentation head 0.001

  \- img_size = 512 x 512

  \- Epoch : 100 -> Early stopping (patience:8)

  \- Batch size : 7

  \- Augmentation : 최종 Augmentation 참고

  \- Optimizer: Adam

  \- Loss: CrossEntropy

  \- Scheduler: StepLR

  \- Conditional Random Fields(CRFs)

  \- Stratified KFold : 김현우 마스터님이 토론게시판에 제공한 5 Fold 사용

  \- 최종 Augmentation으로 진행한 결과

  \- 5 Fold ensemble : 0.6706

  \- 추가 시도

  ​		\- 백본으로 resnext101_32x8d으로 변경하여 시도하였지만 파라미터 개수가 많아 실험 시간 부족.

  ​		\- ReducedLRonPlateau Scheduler로 성능향상이 있었지만 앙상블에 활용하지 못함

  

- **FPN ( backbone: efficientNet b4)**

  \- Learning rate : Encoder 0.001 * 0.1 / Decoder + Segmentation head 0.001

  \- img_size = 512 x 512

  \- Epoch : 100 -> Early stopping (patience:8)

  \- Batch size : 7

  \- Augmentation : 최종 Augmentation 참고

  \- Optimizer : Adam

  \- Loss : CrossEntropy

  \- Scheduler : CosineAnnealingWarmupRestarts

  \- Conditional Random Fields(CRFs)

  \- Stratified KFold : 김현우 마스터님이 토론게시판에 제공한 5 Fold 사용

  \- 최종 Augmentation으로 진행한 결과

  \- 5 Fold ensemble : 0.6621

  

### Ensemble

각각의 Network를 총 5개의 Fold로 학습시킨 결과를 바탕으로 학습된 모델들을 불러와서(총 10개) test dataset의 forward pass를 통해 나온 class별 logit값을 Softmax를 통해 나온 확률의 픽셀별 Soft Voting을 통한 앙상블을 진행했습니다.



### 실험관리

- wandb 사용( 각 실험의 mIoU와 validation되는 이미지들을 확인해가며 진행 )

  ​										![image-20210602144955496](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210602144955496.png)![image-20210602144957649](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210602144957649.png	)

 

### 추가시도

- EfficientUnet( backbone : efficientnet-b4 )
  - 여러가지 backbone, augmentation, optimizer, loss, CRF, finetuning, freeze 사용하여 실험했지만 성능이 위 두 모델보다 떨어짐
  
## ❗ Baseline Appeach

train.py
```
python3 ./train.py --learning_rate=0.0001 --batch_size=8 --nepochs=100 --resize_width=512 --resize_height=512
--patience=5 --seed=42 --num_workers=4 --model="FCN8s" --optimizer="Adam" --criterion="cross_entropy" --scheduler="StepLR"
--train_augmentation="BaseTrainAugmentation" --val_augmentation="BaseTrainAugmentation" --kfold=0 --print_freq=1
--description="Baseline First Trial" --model_save_name="baseline_model.pt"
```

inference.py
```
python3 ./inference.py --model_name="baseline_model.pt"
```

# boostcamp AI Tech - P stage 3

- 대회 기간 : 2021.04.26 ~ 2021.05.07 / 2021.05.10 ~ 2021.05.20
- 팀 : **지구를 지켜조** ([강민용](https://github.com/MignonDeveloper), [김채현](https://github.com/hcworkplace), [노원재](https://github.com/N-analyst), [이정현](https://github.com/gvsteve24), [황정현](https://github.com/rjhwang08))
- 내용 : 재활용 품목 분류를 위한 **Semantic Segmentation** & **Object Detection**



## ❗ 대회 개요 

### 배경

환경 부담을 조금이나마 줄일 수 있는 방법의 하나로 '분리수거'가 있습니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립, 소각되기 때문입니다. 하지만 '이 쓰레기가 어디에 속하는지', '어떤 것들을 분리해서 버리는 것이 맞는지' 등 정확한 분리수거 방법을 알기 어렵다는 문제점이 있습니다.

따라서, 우리는 쓰레기가 찍힌 사진에서 쓰레기를 각각 **Segmentation과 Detection** 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.



--------------------

### 평가 metric

- Semantic Segmentation : **mIoU (Mean Intersection over Union)**
  - 모든 이미지에서 계산된 IoU를 평균내어 반영
- Object Detection : **mAP50 (Mean Average Precision)**
  - 모든 이미지의 각 클래스별 AP 계산 후, 평균내어 반영



## ❗ 데이터 개요

- **전체 이미지 개수** : 4109장
  - train: 2617장 (80%) / public test: 417장 (10%) , private test: 420장(10%)
- **이미지 크기** : (512, 512)
- **11 class** : UNKNOWN, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
  - **Segmentation**에서는 ``Background`` class를 추가하여 총 12개의 class 사용
- **annotation file** : [coco format](https://cocodataset.org/#home) 으로 이루어져 있으며, 다음 2가지 정보를 포함
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

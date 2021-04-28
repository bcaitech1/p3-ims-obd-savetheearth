import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from albumentations import Compose, Resize

import torch
from torch.utils.data import Dataset, DataLoader

from importlib import import_module
import os, random, argparse
from tqdm import tqdm
from pprint import pprint

from recycle_dataset import RecycleDataset

class CFG:
    PROJECT_PATH = "/opt/ml/p3-ims-obd-savetheearth" # 기본 프로젝트 디렉터리
    BASE_DATA_PATH = '/opt/ml/input/data' # 데이터가 저장된 디렉터리
    coco_test_json = 'test.json' # coco annotation test json 파일

    batch_size = 16 # 배치 사이즈
    resize_width = 256 # image resize 가로 크기
    resize_height = 256 # image resize 세로 크기
    seed = 42 # random seed
    num_workers = 4 # 워커의 개수

    model = "Deconvnet_vgg" # model
    test_augmentation = "BaseTestAugmentation"
    model_name = "baseline_model.pt"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # GPU 메모리 사용
    submission_path = 'sample_submission.csv' # submission csv 파일
    docs_path = 'docs'
    model_path = 'models'


# Config Parsing
def get_config():
    parser = argparse.ArgumentParser(description="Mask Classification")

    # Container environment
    parser.add_argument('--PROJECT_PATH', type=str, default=CFG.PROJECT_PATH)
    parser.add_argument('--BASE_DATA_PATH', type=str, default=CFG.BASE_DATA_PATH)
    parser.add_argument('--coco_test_json', type=str, default=CFG.coco_test_json)

    # hyper parameters
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--resize_width", type=int, default=CFG.resize_width, help='resize_width size for image when training')
    parser.add_argument("--resize_height", type=int, default=CFG.resize_height, help='resize_height size for image when training')
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--num_workers", type=int, default=CFG.num_workers)

    # model selection
    parser.add_argument('--model', type=str, default=CFG.model, help=f'model type (default: {CFG.model})')
    parser.add_argument('--test_augmentation', type=str, default=CFG.test_augmentation, help=f'test data augmentation type (default: {CFG.test_augmentation})')
    parser.add_argument("--model_name", type=str, required = True)
    args = parser.parse_args()
    # print(args) # for check arguments
    
    # 키워드 인자로 받은 값을 CFG로 다시 저장합니다.
    CFG.PROJECT_PATH = args.PROJECT_PATH
    CFG.BASE_DATA_PATH = args.BASE_DATA_PATH
    CFG.coco_test_json = args.coco_test_json

    CFG.batch_size = args.batch_size
    CFG.resize_width = args.resize_width
    CFG.resize_height = args.resize_height        
    CFG.seed = args.seed
    CFG.num_workers = args.num_workers

    CFG.model = args.model
    CFG.test_augmentation = args.test_augmentation
    CFG.model_name = args.model_name

    # path setting
    CFG.coco_test_json = os.path.join(CFG.BASE_DATA_PATH, CFG.coco_test_json)
    CFG.submission_path = os.path.join(CFG.PROJECT_PATH, CFG.submission_path) # train csv 파일
    CFG.docs_path = os.path.join(CFG.PROJECT_PATH, CFG.docs_path) # result, visualization 저장 경로
    CFG.model_path = os.path.join(CFG.PROJECT_PATH, CFG.model_path) # trained model 저장 경로

    # for check CFG
    # pprint.pprint(CFG.__dict__) # for check CFG


def set_random_seed():
    # Reproducible Model을 만들기 위해 Random Seed를 고정한다.
    torch.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(CFG.seed)
    random.seed(CFG.seed)
    torch.cuda.manual_seed(CFG.seed)
    torch.cuda.manual_seed_all(CFG.seed) # if use multi-GPU


def get_data_utils():
    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    submission = pd.read_csv(CFG.submission_path)

    test_transform_module = getattr(import_module("augmentation"), CFG.test_augmentation)
    test_transform = test_transform_module()

    test_dataset = RecycleDataset(data_path=CFG.BASE_DATA_PATH,
                                   coco_path=CFG.coco_test_json,
                                   mode='test',
                                   transform=test_transform)

    test_loader = DataLoader(test_dataset,
                           batch_size=CFG.batch_size,
                           shuffle=False,
                           num_workers=CFG.num_workers,
                           collate_fn=collate_fn)

    return submission, test_dataset, test_loader


def get_model():
    model_module = getattr(import_module("recycle_model"), CFG.model)
    model = model_module(num_classes=12)

    model.load_state_dict(torch.load(os.path.join(CFG.model_path, CFG.model_name)))
    model.cuda()

    return model


def inference(model, test_loader, submission):
    print('Start Inference.')
    submission_transform = Compose([Resize(height=CFG.resize_height, width=CFG.resize_width)])
    file_name_list = []
    preds_array = np.empty((0, CFG.resize_width * CFG.resize_height), dtype=np.long)

    model.eval()
    with torch.no_grad():
        for step, (imgs, image_infos) in tqdm(enumerate(test_loader)):
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(CFG.device))
            oms = torch.argmax(outs, dim=1).detach().cpu().numpy()

            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = submission_transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            oms = oms.reshape([oms.shape[0], CFG.resize_width * CFG.resize_height]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            file_name_list.append([i['file_name'] for i in image_infos])

    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    # PredictionString 대입
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(os.path.join(CFG.docs_path, 'results', f'{CFG.model}_{CFG.model_name}.csv'), index=False)


def main():
    # check pytorch version & whether using cuda or not
    print(f"PyTorch version:[{torch.__version__}]")
    print(f"device:[{CFG.device}]")
    print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 개수: {torch.cuda.device_count()}")

    get_config()
    set_random_seed()
    submission, test_dataset, test_loader = get_data_utils()
    model = get_model()
    inference(model, test_loader, submission)


if __name__ == "__main__":
    main()
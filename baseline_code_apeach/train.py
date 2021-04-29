import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import argparse
import cv2
import os, random, pprint, json, time
from tqdm import tqdm
from importlib import import_module


from recycle_dataset import RecycleDataset
from loss import create_criterion
from optimizer import create_optimizer
from scheduler import create_scheduler
from pytorch_tools import EarlyStopping
from utils import label_accuracy_score, add_hist
import wandb

class CFG:
    PROJECT_PATH = "/opt/ml/p3-ims-obd-savetheearth" # 기본 프로젝트 디렉터리
    BASE_DATA_PATH = '/opt/ml/input/data' # 데이터가 저장된 디렉터리
    coco_train_json = 'train.json' # coco annotation train json 파일
    coco_val_json = 'val.json' # coco annotation validation json 파일

    learning_rate = 1e-4 # 러닝 레이트
    batch_size = 16 # 배치 사이즈
    nepochs = 20 # 학습할 에폭수
    resize_width = 512 # image resize 가로 크기
    resize_height = 512 # image resize 세로 크기
    patience = 3 # early stopping을 위한 patience 횟수
    seed = 42 # random seed
    num_workers = 4 # 워커의 개수

    model = "Deconvnet_vgg" # model
    optimizer = "Adam" # optimizer
    criterion = "cross_entropy" # loss function
    scheduler = "StepLR" # learning rate scheduler
    train_augmentation = "BaseAugmentation" # train dataset augmentation
    val_augmentation = "BaseAugmentation" # test dataset augmentation
    kfold = 0 # k-fold
    print_freq = 1 # 결과 출력 빈도
    description = "Baseline First Trial"
    model_save_name = "baseline_model.pt"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # GPU 메모리 사용
    docs_path = 'docs' # result, visualization 저장 경로
    model_path = 'models' # trained model 저장 경로


# Config Parsing
def get_config():
    parser = argparse.ArgumentParser(description="segmentation")

    # Container environment
    parser.add_argument('--PROJECT_PATH', type=str, default=CFG.PROJECT_PATH)
    parser.add_argument('--BASE_DATA_PATH', type=str, default=CFG.BASE_DATA_PATH)
    parser.add_argument('--coco_train_json', type=str, default=CFG.coco_train_json)
    parser.add_argument('--coco_val_json', type=str, default=CFG.coco_val_json)

    # hyper parameters
    parser.add_argument("--learning_rate", type=float, default=CFG.learning_rate, help=f'learning rate (defalut: {CFG.learning_rate})')
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size, help=f'input batch size for training (default: {CFG.batch_size})')
    parser.add_argument("--nepochs", type=int, default=CFG.nepochs, help=f'number of epochs to train (default: {CFG.nepochs})')
    parser.add_argument("--resize_width", type=int, default=CFG.resize_width, help='resize_width size for image when training')
    parser.add_argument("--resize_height", type=int, default=CFG.resize_height, help='resize_height size for image when training')
    parser.add_argument("--patience", type=int, default=CFG.patience, help=f'early stopping patience (default: {CFG.patience})')
    parser.add_argument("--seed", type=int, default=CFG.seed, help=f'random seed (default: {CFG.seed})')
    parser.add_argument("--num_workers", type=int, default=CFG.num_workers, help=f'num workers for data loader (default: {CFG.num_workers})')
    
    # network environment selection
    parser.add_argument('--model', type=str, default=CFG.model, help=f'model type (default: {CFG.model})')
    parser.add_argument('--optimizer', type=str, default=CFG.optimizer, help=f'optimizer type (default: {CFG.optimizer})')
    parser.add_argument('--criterion', type=str, default=CFG.criterion, help=f'criterion type (default: {CFG.criterion})')
    parser.add_argument('--scheduler', type=str, default=CFG.scheduler, help=f'scheduler type (default: {CFG.scheduler})')
    parser.add_argument('--train_augmentation', type=str, default=CFG.train_augmentation, help=f'train data augmentation type (default: {CFG.train_augmentation})')
    parser.add_argument('--val_augmentation', type=str, default=CFG.val_augmentation, help=f'test data augmentation type (default: {CFG.val_augmentation})')
    parser.add_argument('--kfold', type=int, default=CFG.kfold, help=f'K-Fold (default: {CFG.kfold})')
    parser.add_argument("--print_freq", type=int, default=CFG.print_freq, help=f'process print frequency (default: {CFG.print_freq})')
    parser.add_argument('--description', type=str, default=CFG.description, help='model description')
    parser.add_argument('--model_save_name', type=str, default=CFG.model_save_name, help='model save name')

    args = parser.parse_args()
    # print(args) # for check arguments
    
    # 키워드 인자로 받은 값을 CFG로 다시 저장합니다.
    CFG.PROJECT_PATH = args.PROJECT_PATH
    CFG.BASE_DATA_PATH = args.BASE_DATA_PATH
    CFG.coco_train_json = args.coco_train_json
    CFG.coco_val_json = args.coco_val_json

    CFG.learning_rate = args.learning_rate
    CFG.batch_size = args.batch_size
    CFG.nepochs = args.nepochs
    CFG.resize_width = args.resize_width
    CFG.resize_height = args.resize_height      
    CFG.patience = args.patience
    CFG.seed = args.seed
    CFG.num_workers = args.num_workers

    CFG.model = args.model
    CFG.optimizer = args.optimizer
    CFG.criterion = args.criterion
    CFG.scheduler = args.scheduler
    CFG.train_augmentation = args.train_augmentation
    CFG.val_augmentation = args.val_augmentation
    CFG.kfold = args.kfold
    CFG.print_freq = args.print_freq
    CFG.description = args.description
    CFG.model_save_name = args.model_save_name

    CFG.coco_train_json = os.path.join(CFG.BASE_DATA_PATH, CFG.coco_train_json)
    CFG.coco_val_json = os.path.join(CFG.BASE_DATA_PATH, CFG.coco_val_json)
    CFG.docs_path = os.path.join(CFG.PROJECT_PATH, CFG.docs_path)
    CFG.model_path = os.path.join(CFG.PROJECT_PATH, CFG.model_path)
    
    return args
    # for check CFG
    # pprint.pprint(CFG.__dict__) 


def set_random_seed():
    # Reproducible Model을 만들기 위해 Random Seed를 고정한다.
    torch.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(CFG.seed)
    random.seed(CFG.seed)
    torch.cuda.manual_seed(CFG.seed)
    torch.cuda.manual_seed_all(CFG.seed) # if use multi-GPU


def set_logging():
    # for neptune.ai -> experiments 관리
    params = {
        "csv_path": CFG.csv_path,
        "learning_rate": CFG.learning_rate,
        "batch_size": CFG.batch_size,
        "nworkers": CFG.num_workers,
        "nepochs": CFG.nepochs,
        "random_seed": CFG.seed,
        "patience": CFG.patience,
        "resize_width": CFG.resize_width,
        "resize_height": CFG.resize_height,

        "model": CFG.model,
        "kfold": CFG.kfold,
        "train_augmentation": CFG.train_augmentation,
        "val_augmentation": CFG.val_augmentation,
        "optimizer": CFG.optimizer,
        "criterion": CFG.criterion,
        "scheduler": CFG.scheduler,
    }
    run["param"] = params
    run["description"] = CFG.description


def data_visualization():
    pass


# train dataset과 validation dataset에 해당하는 loader를 각각 가져온다.
def get_data_utils():

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # get albumentation transformer from augmentation.py
    train_transform_module = getattr(import_module("augmentation"), CFG.train_augmentation)
    val_transform_module = getattr(import_module("augmentation"), CFG.val_augmentation)
    train_transform = train_transform_module()
    val_transform = val_transform_module()

    train_dataset = RecycleDataset(data_path=CFG.BASE_DATA_PATH,
                                   coco_path=CFG.coco_train_json,
                                   mode='train',
                                   transform=train_transform)
    val_dataset = RecycleDataset(data_path=CFG.BASE_DATA_PATH,
                                 coco_path=CFG.coco_val_json,
                                 mode='val',
                                 transform=val_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=CFG.batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers,
                            collate_fn=collate_fn)

    return train_dataset, val_dataset, train_loader, val_loader


# Defining Model
def get_model():
    # mask_model.py에 정의된 특정 모델을 가져옵니다.
    model_module = getattr(import_module("recycle_model"), CFG.model)
    model = model_module(num_classes=12)

    for count, param in enumerate(model.children()):
        if count < 5:
            param.requires_grad = False

    # 모델의 파라미터를 GPU메모리로 옮깁니다.
    model.cuda()    
    
    # 모델의 파라미터 수를 출력합니다.
    print('parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # GPU가 2개 이상이면 데이터패러럴로 학습 가능하게 만듭니다.
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # loss.py에 정의된 criterion을 가져옵니다.
    criterion = create_criterion('cross_entropy')

    # optimizer.py에 정의된 optimizer를 가져옵니다.
    optimizer = create_optimizer(
        CFG.optimizer,
        params=filter(lambda p: p.requires_grad, model.parameters()),
        # params = model.parameters(),
        lr = CFG.learning_rate,
        weight_decay=1e-6
    )

    # scheduler.py에 정의된 scheduler를 가져옵니다.
    scheduler = create_scheduler(
        CFG.scheduler,
        optimizer=optimizer,
        step_size=20,
        gamma=0.9
    )

    return model, criterion, optimizer, scheduler


# evaluation function for validation data
def func_eval(model, criterion, val_dataset, val_loader):
    best = 0.0 #최고의 평가지표를 가진 모델로 최종 저장하기 위함 #여기서는 best mIoU model
    best_model_wts = copy.deepcopy(model.state_dict())

    print ("Start validation.\n")
    model.eval() # make model evaluation mode

    with torch.no_grad():
        n_class = 12
        total_loss_sum = 0
        mIoU_list = []
        hist = np.zeros((n_class, n_class)) # confusion matrix

        for step, (images, masks, _) in enumerate(val_loader):
            images = torch.stack(images).to(CFG.device)       # (batch, channel, height, width)
            masks = torch.stack(masks).long().to(CFG.device)  # (batch, channel, height, width)
            
            # forward pass
            outputs = model(images)

            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            total_loss_sum += loss.item() * images.shape[0]

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()

            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=n_class)

            del images, masks, outputs

        val_loss = total_loss_sum / len(val_dataset)

    acc, acc_cls, mIoU, iu, fwavacc = label_accuracy_score(hist)
    if best < mIou:
        best = mIoU
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f'==> best model saved - IoU : {best}')

    recycle = ['Background', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    mIoU_df = pd.DataFrame({
        'Recycle Type': recycle,
        'IoU': iu
    })
    return val_loss, acc, mIoU, mIoU_df, best_model_wts

def train(model, criterion, optimizer, scheduler, train_dataset, val_dataset, train_loader, val_loader):
    print ("Start training.\n")
    torch.cuda.empty_cache() #비우고 시작
    early_stopping = EarlyStopping(patience=CFG.patience, path=os.path.join(CFG.model_path, CFG.model_save_name), verbose=True) # early stopping initializing

    for epoch in tqdm(range(CFG.nepochs)):
        model.train() # to train mode
        train_loss_sum = 0

        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images).to(CFG.device)       # (batch, channel, height, width)
            masks = torch.stack(masks).long().to(CFG.device)  # (batch, channel, height, width)
            
            # forward pass
            outputs = model(images)

            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad() # reset gradient 
            loss.backward() # back propagation  
            optimizer.step() # parameters update

            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch, CFG.nepochs, step+1, len(train_loader), loss.item()))
   
            train_loss_sum += loss.item() * images.shape[0]

            del images, masks, outputs

        # scheduler step
        scheduler.step()                 

        # caculate train_loss
        train_loss = train_loss_sum / len(train_dataset)

        # Print #validation
        if ((epoch % CFG.print_freq)==0) or (epoch==(CFG.nepochs - 1)):
            val_loss, acc, mIoU, mIoU_df,best_model_wts = func_eval(model, criterion, val_dataset, val_loader)
            print ("epoch:[%d] train_loss:[%.5f] val_loss:[%.5f] val_mIoU:[%.5f] val_pix_acc: [%.5f]" % (epoch, train_loss, val_loss, mIoU, acc))
            wandb.log({
                "Val Loss": val_loss,
                "Train Loss":train_loss,
                "mIoU" : mIoU,
                "acc" : acc})

            print(mIoU_df)

        # run["epoch/accuracy"].log(accuracy)
        # run["epoch/train_loss"].log(train_loss)
        # run["epoch/val_loss"].log(val_loss)
        # run["epoch/f1_score"].log(f1)

        early_stopping(model=model, mIoU=mIoU)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

    #load best_model(mIoU기준)
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), CFG.model_path+'/best_models' + CFG.model_save_name) #'' : for best_mIoU model saving

    print ("Done")

def main():
    # check pytorch version & whether using cuda or not
    wandb.init()
    wandb.run.name = CFG.description #set wandb run name
    wandb.run.save()
    
    print(f"PyTorch version:[{torch.__version__}]")
    print(f"device:[{CFG.device}]")
    print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 개수: {torch.cuda.device_count()}")

    args = get_config()
    wandb.config.update(args)

    set_random_seed()
    # set_logging()
    # data_visualization(train_df)
    train_dataset, val_dataset, train_loader, val_loader = get_data_utils()
    model, criterion, optimizer, scheduler = get_model()
    wandb.watch(model)
    # func_eval(model, criterion, val_dataset, val_loader)
    train(model, criterion, optimizer, scheduler, train_dataset, val_dataset, train_loader, val_loader)


if __name__ == "__main__":
    main()

"""
python3 ./train.py --learning_rate=0.0001 --batch_size=8 --nepochs=100 --resize_width=512 --resize_height=512\
     --patience=5 --seed=42 --num_workers=4 --model="EffUNet" --optimizer="Adam" --criterion="dicebce" --scheduler="StepLR"\
          --train_augmentation="BaseTrainAugmentation" --val_augmentation="BaseTrainAugmentation" --kfold=0 --print_freq=1\
               --description="effunet 5 Trial" --model_save_name="effunet_5.pt"
"""
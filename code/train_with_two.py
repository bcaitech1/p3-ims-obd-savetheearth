import warnings 
warnings.filterwarnings('ignore')

import wandb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from utils import label_accuracy_score, add_hist, EarlyStopping, dense_crf, dense_crf_wrapper


class CFG:
    PROJECT_PATH = "/opt/ml/save_the_earth" # 기본 프로젝트 디렉터리
    BASE_DATA_PATH = '/opt/ml/input/data' # 데이터가 저장된 디렉터리
    coco_train_json = 'train.json' # coco annotation train json 파일
    coco_val_json = 'val.json' # coco annotation validation json 파일

    learning_rate = 1e-4 # 러닝 레이트
    batch_size = 8 # 배치 사이즈
    valid_batch_size = 24 # validation 배치 사이즈
    nepochs = 20 # 학습할 에폭수
    resize_width = 512 # image resize 가로 크기
    resize_height = 512 # image resize 세로 크기
    patience = 3 # early stopping을 위한 patience 횟수
    seed = 42 # random seed
    num_workers = 4 # 워커의 개수

    model = "FCN8s" # model
    optimizer = "Adam" # optimizer
    criterion = "cross_entropy" # loss function
    scheduler = "StepLR" # learning rate scheduler
    train_augmentation = "BaseAugmentation" # train dataset augmentation
    val_augmentation = "BaseAugmentation" # test dataset augmentation
    kfold = 0 # k-fold
    print_freq = 1 # 결과 출력 빈도
    model_save_name = "baseline_model.pt"
    with_wandb = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # GPU 메모리 사용
    docs_path = 'docs' # result, visualization 저장 경로
    models_path = 'models' # trained model 저장 경로


# Config Parsing
def get_config():
    parser = argparse.ArgumentParser(description="Recycle Segmentation")

    # Container environment

    # hyper parameters
    parser.add_argument("--learning_rate", type=float, default=CFG.learning_rate, help=f'learning rate (defalut: {CFG.learning_rate})')
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size, help=f'input batch size for training (default: {CFG.batch_size})')
    parser.add_argument("--valid_batch_size", type=int, default=CFG.valid_batch_size, help=f'input batch size for validation (default: {CFG.valid_batch_size})')
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
    parser.add_argument('--model_save_name', type=str, default=CFG.model_save_name, help='model save name')

    args = parser.parse_args()
    # print(args) # for check arguments
    
    # 키워드 인자로 받은 값을 CFG로 다시 저장합니다.
    CFG.learning_rate = args.learning_rate
    CFG.batch_size = args.batch_size
    CFG.valid_batch_size = args.valid_batch_size
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
    CFG.model_save_name = args.model_save_name

    CFG.coco_train_json = os.path.join(CFG.BASE_DATA_PATH, CFG.coco_train_json)
    CFG.coco_val_json = os.path.join(CFG.BASE_DATA_PATH, CFG.coco_val_json)
    CFG.docs_path = os.path.join(CFG.PROJECT_PATH, CFG.docs_path)
    CFG.models_path = os.path.join(CFG.PROJECT_PATH, CFG.models_path)
    
    # for check CFG
    # pprint.pprint(CFG.__dict__) 

    wandb.init(project="save_the_earth", entity="mignondev")
    wandb.config.update(args)

def set_random_seed():
    # Reproducible Model을 만들기 위해 Random Seed를 고정한다.
    np.random.seed(CFG.seed)
    random.seed(CFG.seed)
    os.environ['PYTHONHASHSEED'] = str(CFG.seed)
    torch.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(CFG.seed)
    torch.cuda.manual_seed_all(CFG.seed) # if use multi-GPU


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
                              pin_memory=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=CFG.valid_batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers,
                            pin_memory=True,
                            collate_fn=collate_fn)

    return train_dataset, val_dataset, train_loader, val_loader


# Defining Model
def get_model():
    # mask_model.py에 정의된 특정 모델을 가져옵니다.
    model_module = getattr(import_module("recycle_model"), CFG.model)
    model = model_module(num_classes=12)

    # 모델의 파라미터를 GPU메모리로 옮깁니다.
    model.cuda()

    # wandb에서 model 감독
    wandb.watch(model)
    
    # 모델의 파라미터 수를 출력합니다.
    print('parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # GPU가 2개 이상이면 데이터패러럴로 학습 가능하게 만듭니다.
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # loss.py에 정의된 criterion을 가져옵니다.
    criterion = create_criterion(CFG.criterion)

    # optimizer.py에 정의된 optimizer를 가져옵니다.
    optimizer_encoder = create_optimizer(
        CFG.optimizer,
        params=model.seg_model.encoder.parameters(),
        lr = 1e-8
    )

    optimizer_decoder = create_optimizer(
        CFG.optimizer,
        params=[
            {"params": model.seg_model.decoder.parameters()},
            {"params": model.seg_model.segmentation_head.parameters()}
        ],
        lr = 1e-8
    )
    

    # scheduler.py에 정의된 scheduler를 가져옵니다.
    scheduler_encoder = create_scheduler(
        CFG.scheduler,
        optimizer=optimizer_encoder,
        T_0=30,
        T_mult=2,
        eta_max=CFG.learning_rate * 0.1,
        T_up=5,
        gamma=0.3
    )

    scheduler_decoder = create_scheduler(
        CFG.scheduler,
        optimizer=optimizer_decoder,
        T_0=30,
        T_mult=2,
        eta_max=CFG.learning_rate,
        T_up=5,
        gamma=0.3
    )

    return model, criterion, optimizer_encoder, optimizer_decoder, scheduler_encoder, scheduler_decoder


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


# evaluation function for validation data
def func_eval(model, criterion, val_dataset, val_loader, post_crf=False):
    print ("Start validation.\n")
    model.eval() # make model evaluation mode

    with torch.no_grad():
        n_class = 12
        total_loss_sum = 0
        mIoU_list = []
        hist = np.zeros((n_class, n_class)) # confusion matrix

        for step, (images, masks, img_info) in enumerate(val_loader):
            images = torch.stack(images).to(CFG.device)       # (batch, channel, height, width)
            masks = torch.stack(masks).long().to(CFG.device)  # (batch, channel, height, width)
            
            # forward pass
            logits = model(images)

            # loss 계산 (cross entropy loss)
            loss = criterion(logits, masks)
            total_loss_sum += loss.item() * images.shape[0]

            probs = F.softmax(logits, dim=1)
            probs = probs.data.cpu().numpy()

            # Postprocessing
            if post_crf:
                pool = mp.Pool(mp.cpu_count())
                images = images.data.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
                probs = pool.map(dense_crf_wrapper, zip(images, probs))
                pool.close()

            preds = np.argmax(probs, axis=1)
            masks = masks.detach().cpu().numpy()

            hist = add_hist(hist, masks, preds, n_class=n_class)

            if step == 0:
                fig_mask = log_images(masks, preds, img_info)

            del images, masks, logits, probs, preds

        val_loss = total_loss_sum / len(val_dataset)

    acc, acc_cls, mIoU, iu, fwavacc = label_accuracy_score(hist)
    recycle = ['Background', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    mIoU_df = pd.DataFrame({
        'Recycle Type': recycle,
        'IoU': iu
    })
    return val_loss, acc, mIoU, mIoU_df, fig_mask


def train(model, criterion, optimizer_encoder, optimizer_decoder, scheduler_encoder, scheduler_decoder, train_dataset, val_dataset, train_loader, val_loader):
    print ("Start training.\n")

    early_stopping = EarlyStopping(patience=CFG.patience, docs_path=CFG.docs_path, models_path=CFG.models_path, model_name=CFG.model_save_name, verbose=True)

    for epoch in tqdm(range(CFG.nepochs)):
        model.train() # to train mode
        train_loss_sum = 0

        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images).to(CFG.device)       # (batch, channel, height, width)
            masks = torch.stack(masks).long().to(CFG.device)  # (batch, channel, height, width)
            
            # forward pass
            logits = model(images)

            # loss 계산 (cross entropy loss)
            loss = criterion(logits, masks)
            
            optimizer_encoder.zero_grad() # reset gradient 
            optimizer_decoder.zero_grad() # reset gradient 
            loss.backward() # backward propagation
            optimizer_encoder.step() # parameters update
            optimizer_decoder.step() # parameters update

            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'.format(
                    epoch, CFG.nepochs, step+1, len(train_loader), loss.item()))
   
            train_loss_sum += loss.item() * images.shape[0]

            del images, masks, logits

        # scheduler step
        scheduler_encoder.step() 
        scheduler_decoder.step()                      

        # caculate train_loss
        train_loss = train_loss_sum / len(train_dataset)

        # Print
        if ((epoch % CFG.print_freq)==0) or (epoch==(CFG.nepochs - 1)):
            val_loss, acc, mIoU, mIoU_df, fig_mask = func_eval(model, criterion, val_dataset, val_loader)
            print ("epoch:[%d] train_loss:[%.6f] val_loss:[%.6f] val_mIoU:[%.6f] val_pix_acc: [%.6f]" % (epoch, train_loss, val_loss, mIoU, acc))
            print(mIoU_df)

        wandb.log({
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Val mIoU": mIoU,
            "Val pix_acc": acc,
            "Seg": fig_mask,
        })

        metric = dict()
        metric['epoch'] = epoch
        metric['val_loss'] = val_loss
        metric['mIoU'] = mIoU
        metric['mIoU_df'] = mIoU_df
        metric['acc'] = acc

        early_stopping(model=model, mIoU=mIoU, plt=plt, metric=metric, epoch=epoch)
        if early_stopping.early_stop:
            print("Early Stopping")
            print("\n\n\n############################### print best model information ################################\n")
            print(f"epoch : {early_stopping.best_metric['epoch']}")
            print(f"val_loss : {early_stopping.best_metric['val_loss']}")
            print(f"mIoU : {early_stopping.best_metric['mIoU']}")
            print(f"IoU by class : \n {early_stopping.best_metric['mIoU_df']}")
            print(f"acc : {early_stopping.best_metric['acc']}")
            break

        plt.clf()

    print ("Done")


def main():
    # check pytorch version & whether using cuda or not
    print(f"PyTorch version:[{torch.__version__}]")
    print(f"device:[{CFG.device}]")
    print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 개수: {torch.cuda.device_count()}")

    get_config()
    set_random_seed()
    # data_visualization(train_df)
    train_dataset, val_dataset, train_loader, val_loader = get_data_utils()
    model, criterion, optimizer_encoder, optimizer_decoder, scheduler_encoder, scheduler_decoder = get_model()
    # func_eval(model, criterion, val_dataset, val_loader)
    train(model, criterion, optimizer_encoder, optimizer_decoder, scheduler_encoder, scheduler_decoder, train_dataset, val_dataset, train_loader, val_loader)

    wandb.finish()

if __name__ == "__main__":
    main()
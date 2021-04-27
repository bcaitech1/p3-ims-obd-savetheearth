# commit 간단 설명
<br>

## update 0428 daily_mission
- 커밋 해쉬 : [9f65d7d4768141aa694382227c900f9c8fe197b4](https://github.com/bcaitech1/p3-ims-obd-savetheearth/commit/9f65d7d4768141aa694382227c900f9c8fe197b4#diff-217a50de5cf350104657e09374b2a01afda4511754e95a4363e00432814ed4d7)   
- 어피치님 코드 + 모델만 deconvnet으로 바꾼 git 

~~~
python3 ./train.py --learning_rate=0.0001 --batch_size=8 --nepochs=100 --resize_width=512 --resize_height=512 --patience=5 --seed=42 --num_workers=4 --model="Deconvnet_vgg" --optimizer="Adam" --criterion="cross_entropy" --scheduler="StepLR" --train_augmentation="BaseTrainAugmentation" --val_augmentation="BaseTrainAugmentation" --kfold=0 --print_freq=1 --description="Baseline First Trial" --model_save_name="baseline_model.pt"
~~~

## add wandb



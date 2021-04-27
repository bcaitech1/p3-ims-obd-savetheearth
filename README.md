# p3-ims-obd-savetheearth

## Baseline Appeach

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

# DFUC2022

## Evaluation

```
python test.py --rect --modelname lawin --weight /path/to/weight/or/fold --save_path mask_pred --tta v --test_path /path/to/testing/data
```

## Training

```
python train.py --rect --modelname lawin --augmentation --train_path /path/to/training/data
```

Train w/ 5-fold

```
python train.py --rect --modelname lawin --augmentation --train_path /path/to/training/data --kfold 5
```

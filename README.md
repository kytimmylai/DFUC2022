# DFUC2022

<p align="center">
<img src="EnhancedVersion.png" width=100% height=100% 
class="center">
</p>

We propose an accuracy-oriented HarDNet-MSEG, enhancing its backbone and decoder for DFUC.

| Method | DFUC <br> mIoU  |
| :---: |  :---:  | 
| HarDNet-MSEG  | 65.53  | 
| **Enhanced version**  |  **70.63**  | 

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

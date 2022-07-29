# Enhancing Backbone and Decoder of HarDNet-MSEG for Diabetic Foot Ulcer Segmentation
Contains the prediction codes for our submission to the **Foot Ulcer Segmentation Challenge 2022 (DFUC2022)** at **MICCAI2022**.

<p align="center">
<img src="EnhancedVersion.png" width=100% height=100% 
class="center">
</p>

We propose an accuracy-oriented HarDNet-MSEG, enhancing its backbone and decoder for DFUC.

| Method | DFUC mDice |
| :---: |  :---:  | 
| HarDNet-MSEG  | 65.53  | 
| **Enhanced version**  |  **70.63**  | 

## Installation

The related weights are available at https://drive.google.com/drive/folders/1UbuMKLUlCsZAusUVLJqwcBaXiwe0ZUe8?usp=sharing

Download weights and place in the folder ``` /weights ``` 

```
conda create -n dfuc python=3.6
conda activate dfuc
pip install -r requirements.txt
```

## Evaluation

```
python test.py 

Optional Args:
--rect         Padding image before resize to keep its aspect ratio
--tta          Test time augmentation, 'v/h/vh' for verti/horiz/verti&horiz flip
--weight       It can be a weight or a fold. If it's a folder, the result is the mean of each weight result
```

## Training

```
python train.py --rect --augmentation --train_path /path/to/training/data

Optional Args:
--rect         Padding image to square before resize to keep its aspect ratio
--augmentation Activating data audmentation during training
--kfold        Specifying the number of K-Fold Cross-Validation
--k            Training the specific fold of K-Fold Cross-Validation
--dataratio    Specifying the ratio of data spliting for training
--seed         Reproduce the result of a data spliting
```

Train w/ 5-fold

```
python train.py --rect --modelname lawin --augmentation --train_path /path/to/training/data --kfold 5
```

## Our 5 Submission in test phase
1. HarDNet-DFUS & deep1+deep2+boundary loss -- with 5-fold ensemble and vhflip in TTA
```
python test.py --rect --modelname lawinloss4 --tta vh --weight /path/to/five/weight
```
2. HarDNet-DFUS & deep1+deep2+boundary loss -- with 5-fold ensemble and vflip in TTA
```
python test.py --rect --modelname lawinloss4 --tta v --weight /path/to/five/weight
```
3. HarDNet-DFUS & deep1+deep2+boundary loss -- with 5-fold ensemble and hflip in TTA
```
python test.py --rect --modelname lawinloss4 --tta h --weight /path/to/five/weight
```
4. HarDNet-DFUS & deep1+boundary loss -- with 5-fold ensemble and w/o TTA
```
python test.py --rect --modelname lawinloss --weight /path/to/five/weight
```
5. HarDNet-DFUS & deep1+boundary loss -- with 5-fold ensemble and hflip in TTA
```
python test.py --rect --modelname lawinloss --tta h --weight /path/to/five/weight
```


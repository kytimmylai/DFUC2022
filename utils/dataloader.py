# albumentations
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import os
import glob
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.utils import square_padding
from pathlib import Path
from PIL import Image



# +
def calculatemns(img_list, size, rect):
    mean = 0.
    std = 0.
    for name in img_list:
        image = Image.open(name).convert('RGB')
        w, h = image.size
        if rect:
            image = square_padding(image, w, h)
        
        image = transforms.Resize((size, size))(transforms.ToTensor()(image))
        image = image.flatten(1)
        mean += image.mean(1)
        std += image.std(1)

    mean /= len(img_list)
    std /= len(img_list)
    return mean, std


def split_data(length, ratio, k=0, seed=7414, k_fold=1):
    random.seed(seed)
    val_idx = random.sample(range(length), k=length)
    if k_fold == 1:
        val_idx = val_idx[:round(length * (1-ratio))]
    else:
        val_idx = val_idx[(length//k_fold)*k: (length//k_fold)*(k+1)]
    
    train_idx = [x for x in range(length) if x not in val_idx]
    return train_idx, val_idx


# -

class create_dataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, data_path, trainsize, augmentations, train=True, train_ratio=0.8, rect=False, k=0, k_fold=1, seed=None):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.ratio = train_ratio
        self.rect = rect
        try:
            f = []
            for p in data_path if isinstance(data_path, list) else [data_path]:
                #print(p)
                p = Path(p)
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)

            self.images = sorted([x for x in f if ('images' in x) and (x.endswith('.jpg') or x.endswith('.png'))])
            self.gts = sorted([x for x in f if ('masks' in x) and (x.endswith('.jpg') or x.endswith('.png'))])
            length = len(self.images)
            
            #mean, std = calculatemns(self.images, self.trainsize, self.rect)
            #print('mean:', mean, ' std:', std)
            train_idx, val_idx = split_data(length, self.ratio, k=k, seed=seed, k_fold=k_fold)
            
            if train:
                if self.ratio != 1:
                    self.images = sorted([self.images[idx] for idx in train_idx])
                    self.gts = sorted([self.gts[idx] for idx in train_idx])
                
                for i in range(len(self.images)):
                    assert self.images[i].split(os.sep)[-1].split('.')[0] == self.gts[i].split(os.sep)[-1].split('.')[0]
                print('load %g training data from %g images in %s'%(len(self.images), length, data_path))
            
            else:
                if self.ratio != 0:
                    self.images = sorted([self.images[idx] for idx in val_idx])
                    self.gts = sorted([self.gts[idx] for idx in val_idx])
                
                for i in range(len(self.images)):
                    assert self.images[i].split(os.sep)[-1].split('.')[0] == self.gts[i].split(os.sep)[-1].split('.')[0]
                print('load %g validation data from %g images in %s'%(len(self.images), length, data_path))
        
        except Exception as e:
            raise Exception('Error loading data from %s: %s\n' % (data_path, e))
        
        self.size = len(self.images)
        if self.augmentations == True:
            print("data augmentation 2")
            self.transform = A.Compose([
                A.OneOf([
                    A.CenterCrop(480, 480, p=1),
                    A.RandomCrop(480, 480, p=1),
                ], p=0.3),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1 ,scale_limit=0.1, rotate_limit=45, p=0.5),
                A.OneOf([
                    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, min_holes=None, min_height=None, min_width=None, fill_value=0, p=1),
                    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1),
                ], p=0.5)
            ])
   
        else:
            print("no data augmentation")
            self.transform = A.Compose([A.Resize(self.trainsize, self.trainsize)])
        self.nom = transforms.Normalize([0.6262, 0.5736, 0.5512], [0.1561, 0.1819, 0.1938])
        self.totensor = A.Compose([ToTensorV2()])
        
    def __getitem__(self, index):
        # https://github.com/pytorch/vision/issues/9
        seed = np.random.randint(2147483647) # make a seed with numpy generator  #21474

        # Read an image with OpenCV
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)
        name = self.gts[index]
        
        if self.augmentations:
            total = self.transform(image=image, mask=gt)
            image = total["image"]
            gt = total['mask']
            total = A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0, p=0.3)(image=image)
            image = total["image"]
        
        if self.rect:
            if image.shape[0] > image.shape[1]:
                total = A.PadIfNeeded(p=1, min_height=image.shape[0], min_width=image.shape[0], border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)(image=image, mask=gt)
                image = total['image']
                gt = total['mask']
                assert image.shape[0] == image.shape[1], '1, %s, %g/%g' % (self.images[index], image.shape[0], image.shape[1])
            elif image.shape[0] < image.shape[1]:
                total = A.PadIfNeeded(p=1, min_height=image.shape[1], min_width=image.shape[1], border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)(image=image, mask=gt)
                image = total['image']
                gt = total['mask']
                assert image.shape[0] == image.shape[1], '2, %s, %g/%g' % (self.images[index], image.shape[0], image.shape[1])
            else:
                pass
            
        total = A.Resize(self.trainsize, self.trainsize)(image=image, mask=gt)
        image = total["image"]
        gt = total['mask']
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        
        image_final = self.totensor(image=image)
        image = image_final["image"]
        image = self.nom(image)

        gt_final = self.totensor(image=total["mask"], mask=total["mask"])
        gt = gt_final["mask"]
        return image, gt.unsqueeze(0), name

    def __len__(self):
        return self.size


class test_dataset(data.Dataset):
    def __init__(self, data_path, size, rect):
        self.trainsize = size
        try:
            f = []
            for p in data_path if isinstance(data_path, list) else [data_path]:
                p = Path(p)
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            self.images = sorted([x for x in f if 'image' in x])
            length = len(self.images)
            
        except Exception as e:
            raise Exception('Error loading data from %s: %s\n' % (data_path, e))

        print('load %g all images'%length, 'from', data_path)
        mean, std = calculatemns(self.images, self.trainsize, rect)
        print('mean:', mean, ' std:', std)
        
        self.rect = rect
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(), 
            #transforms.Normalize(mean, std)])
            transforms.Normalize([0.6262, 0.5736, 0.5512], [0.1561, 0.1819, 0.1938])])
            
    def __getitem__(self, index):
        name = self.images[index]
        image = Image.open(name).convert('RGB')
        image0 = np.array(image)
        w, h = image.size

        if self.rect:
            image = square_padding(image, w, h)
        
        image = self.transform(image)
        
        return image.unsqueeze(0), name, (h, w), image0

    def __len__(self):
        return self.size

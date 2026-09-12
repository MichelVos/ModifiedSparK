# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Optional, Tuple

import PIL.Image as PImage
import numpy as np
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import glob
try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
    return img


class DANDataset(DatasetFolder):
    def __init__(
            self,
            root_folder: str,
            train: bool,
            transform: Callable,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        root_folder = os.path.join(root_folder, 'train' if train else 'valid')
        super(DANDataset, self).__init__(
            root_folder,
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=None, is_valid_file=is_valid_file
        )
        
        self.samples = tuple(img for (img, label) in self.samples)
        self.targets = None # this is self-supervised learning so we don't need labels
    
    def __getitem__(self, index: int) -> Any:
        img_file_path = self.samples[index]
        return self.transform(self.loader(img_file_path))

class IAMDataset(Dataset):
    def __init__(self, root, transform=None, extensions=(".png", ".jpg", ".jpeg"), samples=None):
        self.paths = []
        self.images = []
        self.mean = None
        self.std = None
        for ext in extensions:
            self.paths.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        self.paths.sort()  # keep consistent order
        '''
        for path in self.paths:
            print(f'Loading image: {path}                   ', end='\r', flush=True)
            img = Image.open(path).convert("RGB")   # grayscale
            self.images.append(img)
            self.mean = np.mean(img) if self.mean is None else self.mean + np.mean(img)
            self.std = np.std(img) if self.std is None else self.std + np.std(img)
        self.mean /= len(self.images)
        self.std /= len(self.images)  
        #self.mean=IMAGENET_DEFAULT_MEAN
        #self.std=IMAGENET_DEFAULT_STD  
        self.mean = self.mean / 255.0 # torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = self.std / 255.0 # torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)        
        '''
        self.mean = np.zeros(3)
        self.std = np.zeros(3)

        for path in self.paths:
            print(f'Loading image: {path}                   ', end='\r', flush=True)

            img = Image.open(path).convert("RGB")
            self.images.append(img)

            img_np = np.array(img, dtype=np.float32)

            self.mean += np.mean(img_np, axis=(0, 1))
            self.std  += np.std(img_np, axis=(0, 1))

        self.mean /= len(self.images)
        self.std  /= len(self.images)

        # convert from [0..255] to [0..1]
        self.mean /= 255.0
        self.std  /= 255.0
        # Option A: ImageNet
        if 'READ' in root:
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]


        self.mean = torch.tensor(self.mean).view(3,1,1).float()
        self.std  = torch.tensor(self.std).view(3,1,1).float()
        print(f'Loaded {len(self.images)} images from {root}, mean={self.mean}, std={self.std}          ')
        self.transform = transform
        self.count = len(self.paths)
        if samples[0] is not None:
            self.count = min(self.count, samples[0])


    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        #path = self.paths[idx]
        #img = Image.open(path).convert("RGB")   # grayscale
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        #normalize image with mean and std computed from the dataset
        img = (img - self.mean) / self.std
        return img # , 0  # dummy label, or replace with real label if you have transcripts


class ImageNetDataset(DatasetFolder):
    def __init__(
            self,
            imagenet_folder: str,
            train: bool,
            transform: Callable,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        imagenet_folder = os.path.join(imagenet_folder, 'train' if train else 'val')
        super(ImageNetDataset, self).__init__(
            imagenet_folder,
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=None, is_valid_file=is_valid_file
        )
        
        self.samples = tuple(img for (img, label) in self.samples)
        self.targets = None # this is self-supervised learning so we don't need labels
    
    def __getitem__(self, index: int) -> Any:
        img_file_path = self.samples[index]
        return self.transform(self.loader(img_file_path))


def build_dataset_to_pretrain(dataset_path, input_size, samples) -> Dataset:
    """
    You may need to modify this function to return your own dataset.
    Define a new class, a subclass of `Dataset`, to replace our ImageNetDataset.
    Use dataset_path to build your image file path list.
    Use input_size to create the transformation function for your images, can refer to the `trans_train` blow. 
    
    :param dataset_path: the folder of dataset
    :param input_size: the input size (image resolution)
    :return: the dataset used for pretraining
    """
    input_size_height, input_size_width = input_size
    trans_train = transforms.Compose([
        transforms.Resize((input_size_height, input_size_width), interpolation=interpolation),
        #transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    
    dataset_path = os.path.abspath(dataset_path)
    for postfix in ('train', 'val'):
        if dataset_path.endswith(postfix):
            dataset_path = dataset_path[:-len(postfix)]
    
    #dataset_train = ImageNetDataset(imagenet_folder=dataset_path, transform=trans_train, train=True)
    dataset_train = IAMDataset(root=dataset_path, transform=trans_train, samples=samples)
    print_transform(trans_train, '[pre-train]')
    return dataset_train

    '''
    def build_dataset_to_pretrain(dataset_path, input_size) -> Dataset:
        """
        You may need to modify this function to return your own dataset.
        Define a new class, a subclass of `Dataset`, to replace our ImageNetDataset.
        Use dataset_path to build your image file path list.
        Use input_size to create the transformation function for your images, can refer to the `trans_train` below. 
        
        :param dataset_path: the folder of dataset
        :param input_size: the input size (image resolution)
        :return: the dataset used for pretraining
        """
        trans_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
        
        dataset_path = os.path.abspath(dataset_path)
        for postfix in ('train', 'val'):
            if dataset_path.endswith(postfix):
                dataset_path = dataset_path[:-len(postfix)]
        
        dataset_train = ImageNetDataset(imagenet_folder=dataset_path, transform=trans_train, train=True)
        print_transform(trans_train, '[pre-train]')
        return dataset_train
    '''

def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')

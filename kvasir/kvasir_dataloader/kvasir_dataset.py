import albumentations as A
import cv2
import random
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def get_label_ID_map():
    '''
    Returns:
        ID2LABEL: list of str, list of label names
        LABEL2ID: dict, mapping from label name to label ID
    '''
    ID2LABEL = ['esophagitis', 'polyps', 'ulcerative-colitis']
    LABEL2ID = {label: i for i, label in enumerate(ID2LABEL)}
    return ID2LABEL, LABEL2ID

class KvasirDataset(Dataset):
    ID2LABEL = ['esophagitis', 'polyps', 'ulcerative-colitis']
    LABEL2ID = {label: i for i, label in enumerate(ID2LABEL)}
    def __init__(self, args, split, transform):
        self.args = args
        self.data_root = args.data_root
        self.split = split
        self.transform = transform
        
        self.data_dir = os.path.join(self.data_root, f'kvasir-dataset-{args.data_version}')
        if not os.path.exists(self.data_dir):
            raise ValueError(f'Invalid data version: {args.data_version}')
        self.image_list_path = os.path.join(
            self.data_root, f'{split}_database-{args.data_version}.txt')
        self.database = self._load_database()

        if self.split == 'train':
            self.A_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RGBShift(p=0.2),
                A.RandomRotate90(p=0.2),
                A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
            ])
        else:
            self.A_transform = A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
            ])

    def __len__(self):
        return len(self.database)
    
    def __getitem__(self, idx):
        image_path, label = self.database[idx]
        image = cv2.imread(os.path.join(self.data_root, image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.A_transform(image=image)['image']
        image = Image.fromarray(image)

        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def _load_database(self):
        image_number_count = [0] * len(self.ID2LABEL)

        database = []
        with open(self.image_list_path, 'r') as f:
            image_list = f.readlines()
            image_list = [line.strip() for line in image_list]
            for image_path in image_list:
                label = self.LABEL2ID[image_path.split('/')[-2]]
                database.append((image_path, label))
                image_number_count[label] += 1
        
        for i, count in enumerate(image_number_count):
            print(f'{self.ID2LABEL[i]}: {count} images')

        return database
    
    def dataset_statistics(self):
        image_number_count = [0] * len(self.ID2LABEL)
        for data in self.database:
            image_path, label = data
            image_number_count[label] += 1
        
        for i, count in enumerate(image_number_count):
            print(f'{self.ID2LABEL[i]}: {count} images')
import albumentations as A
import cv2
import json
import random
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from typing import List, Iterable

from .preprocessing import mask_and_crop_ori_image, min_max_normalization
from .get_list_of_hardest_samples import get_hardest_samples

LIGHT2ID = {'NBI': 0, 'White': 1}
ID2LIGHT = ['NBI', 'White']

def get_label_ID_map(num_classes):
    if num_classes == 4:
        LABEL2ID = {'Normal': 0, 'PCR': 1, 'MPR': 2, 'Cancer': 3}
        ID2LABEL = ['Normal', 'PCR', 'MPR', 'Cancer']
    elif num_classes == 3:
        LABEL2ID = {'Normal': 0, 'PCR': 1, 'MPR': 2, 'Cancer': 2}
        ID2LABEL = ['Normal', 'PCR', 'Cancer']
    elif num_classes == 3.1: # Normal and PCR are combined TODO: fix the code for load image list and load_data in Dataset
        LABEL2ID = {'Normal': 0, 'PCR': 0, 'MPR': 1, 'Cancer': 2}
        ID2LABEL = ['PCR', 'MPR', 'Cancer']
    elif num_classes == 2: # Normal is excluded, only PCR, Cancer (MPR is combined with Cancer)
        LABEL2ID = {'Normal': -1, 'PCR': 0, 'MPR': 1, 'Cancer': 1}
        ID2LABEL = ['PCR', 'Cancer']
    elif num_classes == 2.1: # Normal and PCR are combined
        LABEL2ID = {'Normal': 0, 'PCR': 0, 'MPR': 1, 'Cancer': 1}
        ID2LABEL = ['PCR', 'Cancer']
    else:
        raise ValueError('num_classes should be 2, 3 or 4 instead of {}'.format(num_classes))
    return LABEL2ID, ID2LABEL

class FiveCrossDataset(object):
    def __init__(self, args, transform, fold_cfg='Esophageal-Cancer-Dataset/fold_data.json', patient_info_path='Esophageal-Cancer-Dataset/patient_data_has_normal.json'):
        LABEL2ID, ID2LABEL = get_label_ID_map(args.num_classes)
        self.args = args
        self.data_root = args.data_root
        self.transform = transform
        self.fold_cfg = fold_cfg
        if self.args.num_classes == 2:
            # there are PCR, MPR and Cancer
            patient_info_path = 'Esophageal-Cancer-Dataset/patient_data.json'
            
        with open(patient_info_path, 'r', encoding='utf-8') as f:
            # a dict, key is patient ID, e.g., patient_info['10018690'] = 
            # {'img_id':[], 'img_path':[xxx.jpg], 'label':[['NBI', 'Cancer']], patient_label: 'Cancer'}
            self.patient_info = json.load(f)
            
        with open(fold_cfg, 'r', encoding='utf-8') as f:
            self.fold_info = json.load(f)
            
        with open('Esophageal-Cancer-Dataset/PCR_fold_data.json', 'r', encoding='utf-8') as f:
            self.PCR_fold_info = json.load(f)
    
    def get_fold_image_list(self, fold_idx):
        # e.g., fold_idx = 'fold_1' or 1 
        if isinstance(fold_idx, int):
            fold_idx = 'fold_' + str(fold_idx)
        elif isinstance(fold_idx, str):
            pass
        else:
            raise ValueError('fold_idx should be int or str instead of {}'.format(type(fold_idx)))
        
        assert fold_idx in ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5'], 'fold_idx should be one of fold_1, fold_2, fold_3, fold_4, fold_5 instead of {}'.format(fold_idx)

        image_list = []
        fold_id_list = self.fold_info[fold_idx]
        if self.args.num_classes >= 3:
            for patient_id in fold_id_list:
                image_list.extend(self.patient_info[patient_id]['img_path'])
        elif self.args.num_classes >= 2:
            PCR_fold_id_list = self.PCR_fold_info[fold_idx]
            for patient_id in PCR_fold_id_list:
                image_list.extend(self.patient_info[patient_id]['img_path'])
            for patient_id in fold_id_list:
                patient_label = self.patient_info[patient_id]['patient_label']
                if patient_label == 'PCR':
                    continue
                image_list.extend(self.patient_info[patient_id]['img_path'])
        return image_list
    
    def get_dataloader(self, val_fold_idx):
        print('Dataset for 5-fold cross validation: val fold index:', val_fold_idx)
        train_image_list = []
        test_image_list = self.get_fold_image_list(val_fold_idx)
        for i in range(5):
            if i+1 == val_fold_idx:
                continue
            train_image_list.extend(self.get_fold_image_list(i+1))
        train_image_list, val_image_list = train_test_split(train_image_list, test_size=0.1, random_state=42)

        if self.args.clean_data:
            assert self.args.example_exp_folder is not None, 'example_exp_folder should not be None when "clean_data" is True'
            print('cleaning data using the hardest samples from', self.args.example_exp_folder)
            _, hardest_samples_img_path_list = get_hardest_samples(self.args.example_exp_folder)
            train_image_list = [img_path for img_path in train_image_list if img_path not in hardest_samples_img_path_list]
            val_image_list = [img_path for img_path in val_image_list if img_path not in hardest_samples_img_path_list]
            if self.args.val_data['clean_data']:
                test_image_list = [img_path for img_path in test_image_list if img_path not in hardest_samples_img_path_list]

        train_dataset = EsophagealCancerDataset(self.args, self.transform, train_image_list, split='train')
        val_dataset = EsophagealCancerDataset(self.args, self.transform, val_image_list, split='val')
        test_dataset = EsophagealCancerDataset(self.args, self.transform, test_image_list, split='test')
        
        # dataset statistics
        print('Train dataset statistics:')
        self.dataset_statistics(train_dataset)
        print('Val dataset statistics:')
        self.dataset_statistics(val_dataset)
        print('Test dataset statistics:')
        self.dataset_statistics(test_dataset)
        
        print('Creating dataloader')
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.train_data['batch_size'], shuffle=True, num_workers=self.args.train_data['num_workers'])
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.train_data['batch_size']//2, shuffle=False, num_workers=self.args.train_data['num_workers']//2)
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.val_data['batch_size'], shuffle=False, num_workers=self.args.val_data['num_workers'])
        assert self.args.val_data['batch_size'] == 1, 'batch size of val set should be 1 for filename and patient_id.'
        return train_dataloader, val_dataloader, test_dataloader
    
    def dataset_statistics(self, dataset):
        # count the number of images of each class
        LABEL2ID, ID2LABEL = get_label_ID_map(dataset.args.num_classes)
        count = [0, 0, 0, 0]
        for d in dataset.database:
            label = d['img_label']
            count[label] += 1
        for i in range(len(ID2LABEL)):
            print('    {}: {}'.format(ID2LABEL[i], count[i]))

class EsophagealCancerDataset(torch.utils.data.Dataset):
    '''
    transform comes from timm.data.transforms_factory.create_transform,
    so its input should be an image from PIL.Image.open.
    But albumentations.Compose can only accept numpy.ndarray as input, 
    and the color channel should be in the last dimension as RGB. (so if use cv2.imread, remember to convert BGR to RGB)
    Here we firstly read the image using Image.open, then convert it to numpy.ndarray.
    '''
    def __init__(self, args, transform, image_list, split='train'):
        self.args = args
        self.data_root = args.data_root
        try:
            self.ood_data_root = args.ood_data_root
        except:
            self.ood_data_root = 'hard_samples'
        print('ood_data_root:', self.ood_data_root)
        self.croped_data_root = 'Esophageal-Cancer-Dataset/Aug_Dataset'
        self.mask_root = 'Esophageal-Cancer-Dataset/Mask'
        self.transform = transform
        assert self.transform is not None, 'transform should not be None'
        self.image_list = image_list
        self.split = split
        if self.split == 'train':
            if self.args.train_data['data_augmentation']:
                self.A_transform = A.Compose([
                    A.RandomResizedCrop(height=self.args.input_size, width=self.args.input_size, scale=(0.8, 0.8), ratio=(1.0, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.1),
                    A.RandomRotate90(p=0.5),
                    # A.RandomToneCurve(p=0.2),
                ])
            else:
                self.A_transform = A.Compose([
                    A.RandomResizedCrop(height=self.args.input_size, width=self.args.input_size, scale=(0.8, 0.8), ratio=(1.0, 1.0))
                ])
        elif self.split == 'val' or self.split == 'test':
            if self.args.val_data['crop_val']:
                self.A_transform = A.Compose([A.RandomResizedCrop(height=self.args.input_size, width=self.args.input_size, scale=(0.8, 0.8), ratio=(1.0, 1.0))])
            else:
                self.A_transform = None
        else:
            raise ValueError('split should be train, val or test instead of {}'.format(self.split))

        if self.split != 'train':
            self.crop_val = self.args.val_data['crop_val']
        else:
            self.crop_val = True # default crop_val for val set is True
        
        self.database = self._load_database()

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        data = self.database[idx]
        filename = data['filename']
        croped_image = self.read_image(filename)
        croped_image = self.transform(croped_image) # tensor
        # remember that the image after pytorch normalization is not in the range (0-1) !!!
        # croped_image = min_max_normalization(croped_image)
        label = torch.tensor(data['img_label'], dtype=torch.long)
        patient_id = torch.tensor(data['patient_id'], dtype=torch.long)
        
        # check whether filename in the hard image folder, if so, label it as ood data
        # "Esophageal-Cancer-Dataset/Cancer/NBI-Cancer/10018690_A25178AA.jpg"
        check_file_path = filename.replace('Esophageal-Cancer-Dataset', self.ood_data_root)
        if os.path.exists(check_file_path):
            ood_label = torch.tensor(int(self.args.num_classes), dtype=torch.long)
        else:
            ood_label = label

        if self.split == 'train':
            return croped_image, label, ood_label
        else:
            return croped_image, label, filename, patient_id, ood_label
    
    def read_image(self, filename):
        
        base_name = os.path.basename(filename)
        croped_image_path = os.path.join(self.croped_data_root, base_name.replace('.jpg', '.png'))
        
        if os.path.exists(croped_image_path):
            croped_image = cv2.imread(croped_image_path, cv2.IMREAD_COLOR)
            croped_image = cv2.cvtColor(croped_image, cv2.COLOR_BGR2RGB)
        else:
            maskfilename = os.path.join(self.mask_root, os.path.basename(filename))
            ori_image = cv2.imread(filename, cv2.IMREAD_COLOR)
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(maskfilename, cv2.IMREAD_COLOR)
            croped_image = mask_and_crop_ori_image(ori_image, mask) # shape: (512, 512, 3) cv2 format RGB
            ### save the croped image for speed up
            os.makedirs(os.path.dirname(croped_image_path), exist_ok=True)
            cv2.imwrite(croped_image_path, cv2.cvtColor(croped_image, cv2.COLOR_RGB2BGR))
        
        if self.A_transform is not None:
            croped_image = self.A_transform(image=croped_image)['image']
        
        croped_image = Image.fromarray(croped_image)
        return croped_image
        
    def _load_database(self):
        # load data from self.data_root
        LABEL2ID, ID2LABEL = get_label_ID_map(self.args.num_classes)
        ### code for class balance
        if self.split == 'train' or self.split == 'val':
            balanced = self.args.train_data['balanced']
        else:
            balanced = self.args.val_data['balanced']
        
        database = []
        filtered_image_list = []
        for filename in self.image_list:
            patient_id, light_type, img_label, patient_label = self.get_ID_and_label_from_filename(filename)
            if light_type not in self.args.imaging_type:
                continue
            patient_id = int(patient_id)
            light_type = LIGHT2ID[light_type]
            img_label = LABEL2ID[img_label]
            if img_label < 0:
                continue
            
            try:
                if self.args.no_MPR and ID2LABEL[img_label] == 'MPR':
                    continue
            except:
                pass
            
            if balanced:
                p = random.random()
                if ID2LABEL[img_label] != 'PCR' and p > 0.3:
                    continue
                
            ###
            patient_label = LABEL2ID[patient_label]
            data = {
                'patient_id': patient_id,
                'light_type': light_type,
                'img_label': img_label,
                'patient_label': patient_label,
                'filename': filename
            }

            if balanced and ID2LABEL[img_label] == 'MPR':
                raise NotImplementedError('augmentation for MPR is not implemented yet')
            
            database.append(data)
            filtered_image_list.append(filename)
        
        self.image_list = filtered_image_list
        return tuple(database)
    
    @staticmethod
    def get_ID_and_label_from_filename(filename):
        # "Esophageal-Cancer-Dataset/Cancer/NBI-Cancer/10018690_A25178AA.jpg"
        patient_id = filename.split('/')[-1].split('_')[0]
        light_type, img_label = filename.split('/')[-2].split('-')
        patient_label = filename.split('/')[-3]
        return patient_id, light_type, img_label, patient_label
        
    def __add__(self, other_dataset):
        return EsophagealCancerDataset(self.args, self.transform, self.image_list + other_dataset.image_list)
        

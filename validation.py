from pytorch_attribution import GradCAM, GradCAMPlusPlus, XGradCAM, AblationCAM, GuidedBackProp, IntegratedGradients, GuidedIG, CombinedWrapper, EigenGradCAM, LayerCAM, GuidedGradCAM, Occlusion, FullGrad, DFA
from pytorch_attribution import get_reshape_transform, normalize_saliency, visualize_single_saliency

import argparse
import cv2
import os
from sklearn.metrics import confusion_matrix, classification_report
import sys
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
from tqdm import tqdm
import wandb
import yaml
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile

from data import get_label_ID_map, EsophagealCancerDataset
from generate_test_report import generate_cross_validation_report
from model import get_timm_model, get_trained_model, get_local_timm_checkpoint_path
from utils import parse_args_and_yaml, set_seed, make_exp_folder, Logger
from data import get_hardest_samples

def get_hard_image_path_list(log_root):
    wrong_cnt = {}
    for file in os.listdir(log_root):
        if file.endswith('.txt'):
            with open(os.path.join(log_root, file), 'r') as f:
                for line in f:
                    filename, count = line.strip().split()
                    if filename not in wrong_cnt.keys():
                        wrong_cnt[filename] = 0
                    wrong_cnt[filename] += int(count)
    wrong_cnt = sorted(wrong_cnt.items(), key=lambda x: x[1], reverse=True)
    # save the wrong list
    #with open(os.path.join(log_root, 'wrong_list.txt'), 'w') as f:
    #    for filename, count in wrong_cnt:
    #        f.write(f'{filename} {count}\n')
    return wrong_cnt

def get_image_path_list(data_dir):
    log_root = 'validation_output'
    wrong_cnt = get_hard_image_path_list(log_root)
    hard_list = [filename for filename, count in wrong_cnt if count >= 40]
    
    image_path_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path_list.append(os.path.join(root, file))
    print(len(image_path_list), image_path_list[:2])
    image_path_list = list(set(image_path_list) - set(hard_list))
    print(len(image_path_list), image_path_list[:2])
    
    return image_path_list

set_seed(42)

if __name__ == '__main__':
    data_dir = 'Validation-Dataset'
    image_path_list = get_image_path_list(data_dir)

    # args
    parser = argparse.ArgumentParser(description='post-hoc attribution methods for debugging models')
    parser.add_argument('--exp_path', type=str, default='', help='exp_folder_path', metavar='FILE')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id')
    vis_save_folder = os.path.join(parser.parse_args().exp_path, 'vis')
    if not os.path.exists(vis_save_folder):
        os.makedirs(vis_save_folder)


    gpu = parser.parse_args().gpu
    args_path = os.path.join(parser.parse_args().exp_path, 'args.yaml')
    with open(args_path, 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f)
        parser.set_defaults(**args)
    args = parser.parse_args()
    args.gpu_id = args.gpu = gpu
    args.data_root = 'Validation-Dataset'
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)


    LABEL2ID, ID2LABEL = get_label_ID_map(args.num_classes)
    
    
    ckpt_path_template = os.path.join(args.exp_path, 'fold{}_{}_best.pth')
    model, model_config, transform = get_timm_model(args.model_name, num_classes=len(ID2LABEL), pretrained=False, args=args)
    
    # used for denormalization and visualization
    mean = model_config['mean']
    std = model_config['std']
    mean = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()
    
    # Dataset
    dataset = EsophagealCancerDataset(args=args, transform=transform, image_list=image_path_list, split='test')
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    log_folder = 'validation_output'
    log_path = os.path.join(log_folder, args.model_name + '.log')
    # save ckpt path
    with open(log_path, 'a') as f:
        f.write(args.exp_path + '\n')

    @torch.no_grad()
    def test_one_fold_model(model, fold, dataloader=test_dataloader):
        ckpt_path = ckpt_path_template.format(fold, args.model_name)
        print(ckpt_path)
        # load model
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
        model.eval()
        model = model.cuda()
        
        
        pred_list = []
        target_list = []
        wrong_filename_list = []
        acc = 0.
        for (img, label, filename, patient_id, ood_label) in tqdm(dataloader):
            output = model(img.cuda())
            pred = output.argmax(dim=1).cpu().item()
            
            pred_list.append(pred)
            target_list.append(label.item())
            
            
            if pred != label:
                wrong_filename_list.append(filename[0])
            else:
                acc += 1
        acc = acc / len(dataloader.dataset)
        print(f'acc: {acc:.4f}')
        return pred_list, target_list, wrong_filename_list

    pred_list, target_list = [], []
    wrong_filename_list = {}
    for fold_idx in range(1, 6):
        
        one_fold_pred, one_fold_target, one_fold_wrong_filename_list = test_one_fold_model(model, fold_idx)
        pred_list.extend(one_fold_pred)
        target_list.extend(one_fold_target)
        for filename in one_fold_wrong_filename_list:
            if filename not in wrong_filename_list.keys():
                wrong_filename_list[filename] = 0
            wrong_filename_list[filename] += 1
        print('==================================')
    
    wrong_filename_list = sorted(wrong_filename_list.items(), key=lambda x: x[1], reverse=True)
        
    # save results
    with open(log_path, 'a') as f:    
        f.write(f'Classification Report:\n{classification_report(target_list, pred_list, target_names=ID2LABEL, digits=4)}\n')
        f.write(f'Confusion Matrix:\n{confusion_matrix(target_list, pred_list)}\n')

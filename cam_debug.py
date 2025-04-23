from pytorch_attribution import GradCAM, GradCAMPlusPlus, XGradCAM, AblationCAM, GuidedBackProp, IntegratedGradients, VanillaGradient
from pytorch_attribution import GuidedIG, CombinedWrapper, EigenGradCAM, LayerCAM, GuidedGradCAM, Occlusion, FullGrad, DFA, BlurIG
from pytorch_attribution import get_reshape_transform, normalize_saliency, visualize_single_saliency

import argparse
import cv2
import os
from sklearn.metrics import confusion_matrix, classification_report
import sys
import time
import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
from tqdm import tqdm
import wandb
import yaml
import matplotlib.pyplot as plt
import numpy as np

from data import FiveCrossDataset, get_label_ID_map
from generate_test_report import generate_cross_validation_report
from model import get_timm_model, get_trained_model, get_local_timm_checkpoint_path
from utils import parse_args_and_yaml, set_seed, make_exp_folder, Logger
from data import get_hardest_samples, EsophagealCancerDataset

from main import test

global LABEL2ID, ID2LABEL

def get_hard_image_path_list(log_root):
    wrong_cnt = {}
    for file in os.listdir(log_root):
        if file.endswith('.txt') and file != 'wrong_list.txt':
            with open(os.path.join(log_root, file), 'r') as f:
                for line in f:
                    filename, count = line.strip().split()
                    if filename not in wrong_cnt.keys():
                        wrong_cnt[filename] = 0
                    wrong_cnt[filename] += int(count)
    wrong_cnt = sorted(wrong_cnt.items(), key=lambda x: x[1], reverse=True)
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

# fix random seed
set_seed(42)

if __name__ == '__main__':
    # add environment variable for huggingface mirror
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print('HF_ENDPOINT:', os.environ['HF_ENDPOINT'])

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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    FOLD_ID = 1

    LABEL2ID, ID2LABEL = get_label_ID_map(args.num_classes)
    ckpt_path = os.path.join(args.exp_path, f'fold{FOLD_ID}_{args.model_name}_best.pth')
    model, model_config, transform = get_timm_model(args.model_name, num_classes=len(ID2LABEL), pretrained=False, args=args)
    mean = model_config['mean']
    std = model_config['std']
    mean = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()
    
    
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
    model.eval()
    model = model.cuda()
    
    for name, module in model.named_modules():
        print(name)
    
    if 'swin' in args.model_name:
        reshape_transform = get_reshape_transform(False)
    elif 'vit' in args.model_name:
        reshape_transform = get_reshape_transform(True)
    else:
        reshape_transform = None

    attribution_net = XGradCAM(model, reshape_transform=reshape_transform)
    # attribution_net = BlurIG(model)
    # attribution_net = CombinedWrapper(model, IntegratedGradients, XGradCAM)
    kwargs_for_gradient_net = {
        'steps': 50,
        'max_sigma': 100,
        'grad_step': 0.01,
        'sqrt': False,
        'batch_size': 4,
        #'fraction': 0.25,
        #'max_dist': 0.02
    }
    target_layer_candidates = list()
    for name, module in model.named_modules():
        print(name)
        target_layer_candidates.append(name)

    target_layer = input('Enter the target layer: ')
    while target_layer not in target_layer_candidates:
        print('Invalid layer name')
        target_layer = input('Enter the target layer: ')
    attribution_net = attribution_net.cuda()

    vis_save_folder = os.path.join(parser.parse_args().exp_path, f'vis_{target_layer}')
    if not os.path.exists(vis_save_folder):
        os.makedirs(vis_save_folder)

    '''args.val_data['clean_data'] = False
    # args.val_data['balanced'] = True
    corss_val_data_solver = FiveCrossDataset(args, transform)
    _, _, test_loader = corss_val_data_solver.get_dataloader(FOLD_ID)'''
    
    data_dir = 'Validation-Dataset'
    image_path_list = get_image_path_list(data_dir)
    test_dataset = EsophagealCancerDataset(args, transform, image_path_list, split='test')

    
    pred_list = []
    target_list = []
    filename_list = []
    wrong_pred_filename_list = []
    acc = 0.
    pcr_right_cnt = 0
    cancer_right_cnt = 0
    #for (img, label, filename, patient_id, ood_label) in tqdm(test_loader.dataset):
    for (img, label, filename, patient_id, ood_label) in tqdm(test_dataset):
        with torch.no_grad():
            img = img.unsqueeze(0).cuda()
            label = torch.tensor([label]).cuda()
            filename_list.append(filename)
            
            
            output = attribution_net(img)
            pred = output.argmax(dim=1)
            pred_list.extend(pred.cpu().tolist())
            target_list.extend(label.cpu().tolist())
            prob = torch.nn.functional.softmax(output, dim=1)
        if pred != label:    
            prob_title = f'{ID2LABEL[label.item()]}: {prob[0][label].item():.4f}, {ID2LABEL[pred.item()]}: {prob[0][pred].item():.4f}'
            wrong_pred_filename_list.append([filename, ID2LABEL[label.item()], ID2LABEL[pred.item()]])
            target_index = label
            if isinstance(attribution_net, CombinedWrapper):
                target_index_attribution = attribution_net.get_mask(img, target_index, target_layer=target_layer, **kwargs_for_gradient_net)
            elif isinstance(attribution_net, VanillaGradient):
                target_index_attribution = attribution_net.get_mask(img, target_index, **kwargs_for_gradient_net)
            else:
                target_index_attribution = attribution_net.get_mask(img, target_index, target_layer=target_layer)
            target_index_attribution = normalize_saliency(target_index_attribution)
            target_index_attribution = visualize_single_saliency(target_index_attribution[0].unsqueeze(0))
            
            pred_index = pred
            if isinstance(attribution_net, CombinedWrapper):
                pred_index_attribution = attribution_net.get_mask(img, pred_index, target_layer=target_layer, **kwargs_for_gradient_net)
            elif isinstance(attribution_net, VanillaGradient):
                pred_index_attribution = attribution_net.get_mask(img, pred_index, **kwargs_for_gradient_net)
            else:
                pred_index_attribution = attribution_net.get_mask(img, pred_index, target_layer=target_layer)
            pred_index_attribution = normalize_saliency(pred_index_attribution)
            pred_index_attribution = visualize_single_saliency(pred_index_attribution[0].unsqueeze(0))
            
            # denormalize img
            img = img * std + mean
            img = img[0].permute(1, 2, 0).cpu().numpy()
            img = np.uint8(img * 255)
            
            # get two overlapped images
            img_target = cv2.addWeighted(img, 0.5, target_index_attribution, 0.5, 0)
            img_pred = cv2.addWeighted(img, 0.5, pred_index_attribution, 0.5, 0)
            
            
            vis_name = filename.split('/')[-2]
            save_id = filename.split('/')[-1].split('.')[0]
            
            plt.figure(figsize=(16, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title(f'{vis_name}, {prob_title}')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(img_target)
            plt.title(f'target: {ID2LABEL[label.item()]}')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(img_pred)
            plt.title(f'pred: {ID2LABEL[pred.item()]}')
            plt.axis('off')
            plt.tight_layout()
            plt.axis('off')
            plt.savefig(os.path.join(vis_save_folder, 'wrong' + save_id + '.png'))
            plt.close()
            
            os.makedirs(os.path.join(vis_save_folder, save_id), exist_ok=True)
            cv2.imwrite(os.path.join(vis_save_folder, save_id, 'original.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(vis_save_folder, save_id, f'{ID2LABEL[target_list[-1]]}.png'), cv2.cvtColor(img_target, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(vis_save_folder, save_id, f'{ID2LABEL[pred_list[-1]]}.png'), cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR))
            
        else:
            if pred == label:
                acc += 1            
            
            vis_name = filename.split('/')[-2]
            save_id = filename.split('/')[-1].split('.')[0]
            pcr_index = LABEL2ID['PCR']
            if isinstance(attribution_net, CombinedWrapper):
                pcr_index_attribution = attribution_net.get_mask(img, pcr_index, target_layer=target_layer, **kwargs_for_gradient_net)
            elif isinstance(attribution_net, VanillaGradient):
                pcr_index_attribution = attribution_net.get_mask(img, pcr_index, **kwargs_for_gradient_net)
            else:
                pcr_index_attribution = attribution_net.get_mask(img, pcr_index, target_layer=target_layer)
            pcr_index_attribution = normalize_saliency(pcr_index_attribution)
            pcr_index_attribution = visualize_single_saliency(pcr_index_attribution[0].unsqueeze(0))
            cancer_index = LABEL2ID['Cancer']
            if isinstance(attribution_net, CombinedWrapper):
                cancer_index_attribution = attribution_net.get_mask(img, cancer_index, target_layer=target_layer, **kwargs_for_gradient_net)
            elif isinstance(attribution_net, VanillaGradient):
                cancer_index_attribution = attribution_net.get_mask(img, cancer_index, **kwargs_for_gradient_net)
            else:
                cancer_index_attribution = attribution_net.get_mask(img, cancer_index, target_layer=target_layer)
            cancer_index_attribution = normalize_saliency(cancer_index_attribution)
            cancer_index_attribution = visualize_single_saliency(cancer_index_attribution[0].unsqueeze(0))
            
            # denormalize img
            img = img * std + mean
            img = img[0].permute(1, 2, 0).cpu().numpy()
            img = np.uint8(img * 255)
            
            image_pcr = cv2.addWeighted(img, 0.5, pcr_index_attribution, 0.5, 0)
            image_cancer = cv2.addWeighted(img, 0.5, cancer_index_attribution, 0.5, 0)
            
            plt.figure(figsize=(16, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title(f'{vis_name} {ID2LABEL[label.item()]}')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(image_pcr)
            plt.title(f'PCR: {prob[0][pcr_index].item():.4f}')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(image_cancer)
            plt.title(f'Cancer: {prob[0][cancer_index].item():.4f}')
            plt.axis('off')
            plt.tight_layout()
            if label == pred:
                plt.savefig(os.path.join(vis_save_folder, 'right' + save_id + '.png'))
            else:
                plt.savefig(os.path.join(vis_save_folder, 'wrong' + save_id + '.png'))
            plt.close()
            os.makedirs(os.path.join(vis_save_folder, save_id), exist_ok=True)
            # 分别保存原图
            cv2.imwrite(os.path.join(vis_save_folder, save_id, 'original.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(vis_save_folder, save_id, 'PCR.png'), cv2.cvtColor(image_pcr, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(vis_save_folder, save_id, 'Cancer.png'), cv2.cvtColor(image_cancer, cv2.COLOR_RGB2BGR))
            
            
    # acc = acc / len(test_loader.dataset)
    acc = acc / len(test_dataset)
    print(f'acc: {acc:.4f}')
    
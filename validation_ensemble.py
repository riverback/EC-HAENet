from pytorch_attribution import GradCAM, GradCAMPlusPlus, XGradCAM, AblationCAM, GuidedBackProp, IntegratedGradients, GuidedIG, CombinedWrapper, EigenGradCAM, LayerCAM, GuidedGradCAM, Occlusion, FullGrad, DFA
from pytorch_attribution import get_reshape_transform, normalize_saliency, visualize_single_saliency

import argparse
import cv2
import os
from sklearn.metrics import confusion_matrix, classification_report
import sys
import time
import torch
from torch import nn
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

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from data import get_label_ID_map, EsophagealCancerDataset, FiveCrossDataset
from generate_test_report import generate_cross_validation_report
from model import get_timm_model, get_trained_model, get_local_timm_checkpoint_path
from utils import parse_args_and_yaml, set_seed, make_exp_folder, Logger
from data import get_hardest_samples
from confidence_interval import generate_CI


def get_image_path_list(data_dir):
    log_root = 'validation_output'
    
    
    image_path_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path_list.append(os.path.join(root, file))
    print(len(image_path_list), image_path_list[:2])

    # remove non-MPR
    new_image_path_list = []
    for i in range(len(image_path_list)):
        if 'White-MPR' in image_path_list[i]:
            new_image_path_list.append(image_path_list[i])
    image_path_list = new_image_path_list

    print(len(image_path_list), image_path_list[:2])
    

    return image_path_list

set_seed(42)


class EnsembleModel(nn.Module):
    def __init__(self, models, ensemble_type='soft'):
        super(EnsembleModel, self).__init__()
        assert ensemble_type in ['soft', 'hard', 'stacking'], 'ensemble type should be soft, hard or stacking'
        self.models = models
        self.ensemble_type = ensemble_type
    
    def forward(self, x):
        if self.ensemble_type == 'hard':
            outputs = [m(x).argmax(dim=1).cpu().item() for m in self.models]
            pred = max(set(outputs), key=outputs.count)
            prob = 1.0 * outputs.count(pred) / len(outputs)
            return pred, prob
        elif self.ensemble_type == 'soft':
            outputs = [m(x) for m in self.models]
            outputs = torch.stack(outputs, dim=1)
            outputs = outputs.mean(dim=1)
            prob = nn.functional.softmax(outputs, dim=1).detach().cpu().squeeze().numpy().tolist()
            pred = outputs.argmax(dim=1).cpu().item()
            return pred, prob[pred]
        else:
            raise ValueError('stacking not implemented yet')
        
class ECHAENet(nn.Module):
    '''a new classifier on top of the ensemble of models'''
    def __init__(self, models, num_classes=2):
        super(ECHAENet, self).__init__()
        self.models = models
        self.num_classes = num_classes
        feature_dim = 0
        test_input_shape = (1, 3, 224, 224)
        for model in self.models:
            model.eval()
            model = model.cuda()
            with torch.no_grad():
                x = torch.randn(test_input_shape).cuda()
                feature = model(x)
                feature_dim += feature.shape[1]
        self.fc = nn.Linear(feature_dim, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.ensemble_type = 'soft'

    def forward(self, x):
        outputs = [m(x) for m in self.models]
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.mean(dim=1)
        prob = nn.functional.softmax(outputs, dim=1).detach().cpu().squeeze().numpy().tolist()
        pred = outputs.argmax(dim=1).cpu().item()
        return pred, prob[pred]

@torch.no_grad()
def main_ensemble_one_type(exp_path, gpu, ensemble_type='soft'):
    data_dir = 'Validation-Dataset'
    image_path_list = get_image_path_list(data_dir)

    # args
    parser = argparse.ArgumentParser(description='post-hoc attribution methods for debugging models')
    args_path = os.path.join(exp_path, 'args.yaml')
    with open(args_path, 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f)
        parser.set_defaults(**args)
    args = parser.parse_args()
    
    args.gpu_id = args.gpu = gpu
    args.data_root = 'Validation-Dataset'
    args.exp_path = exp_path
    
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
    log_path = os.path.join(log_folder, 'ensemble-'+args.model_name + '.log')
    log_path = os.path.join(log_folder, 'ensemble-many' + '.log')
    # save ckpt path
    with open(log_path, 'a') as f:
        f.write(args.exp_path + '\n')

    model_list = []
    for i in range(1, 6):
        model, model_config, transform = get_timm_model(args.model_name, num_classes=len(ID2LABEL), pretrained=False, args=args)
        model_path = ckpt_path_template.format(i, args.model_name)
        model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
        model.eval()
        model = model.cuda()
        model_list.append(model)
    
    model = EnsembleModel(model_list, ensemble_type)
    model.eval()
    model = model.cuda()
    
    def test_ensemble(model, dataloader=test_dataloader):
        pred_list = []
        prob_list = []
        label_list = []
        target_list = []
        wrong_filename_list = []
        acc = 0
        for (img, label, filename, patient_id, ood_label) in tqdm(dataloader):
            pred, prob = model(img.cuda())
            # print(output_list, pred)
            pred_list.append(pred)
            target_list.append(label.item())
            
            if pred != label:
                wrong_filename_list.append(filename[0])
            else:
                acc += 1
                
            if label == 0:
                label_list.append(1)
                if pred == label:
                    prob_list.append(prob)
                else:
                    prob_list.append(1 - prob)
            else:
                label_list.append(-1)
                if pred == label:
                    prob_list.append(1 - prob)
                else:
                    prob_list.append(prob)
                    
        print(f'accuracy: {acc / len(dataloader)}')
                
        return pred_list, target_list, prob_list, wrong_filename_list, label_list

    pred_list, target_list, prob_list, wrong_filename_list, label_list = test_ensemble(model)
    
    # save results
    with open(log_path, 'a') as f:    
        f.write(f'####ensembling strategy: {model.ensemble_type} ####\n')
        f.write(f'Classification Report:\n{classification_report(target_list, pred_list, target_names=ID2LABEL, digits=4)}\n')
        f.write(f'Confusion Matrix:\n{confusion_matrix(target_list, pred_list)}\n')
    
    # plot roc curve
    display = RocCurveDisplay.from_predictions(
        y_true=np.array(label_list),
        y_pred=np.array(prob_list),
        color="darkorange",
        name="ECNet",
        plot_chance_level=True,
    )
    _ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    )
    plt.savefig('roc_curve.png')
    prob_log_path = os.path.join(log_folder, 'ensemble-'+args.model_name + '_prob.csv')
    with open(prob_log_path, 'w') as f:
        for prob, label in zip(prob_list, label_list):
            f.write(f'{prob, label}\n')


@torch.no_grad()
def main_ensemble_different_type(exp_path_list, gpu, ensemble_type='soft'):
    parser = argparse.ArgumentParser(description='post-hoc attribution methods for debugging models')
    
    data_dir = 'Validation-Dataset'
    image_path_list = get_image_path_list(data_dir)


    model_list = []
    for exp_path in exp_path_list:
        args_path = os.path.join(exp_path, 'args.yaml')
        with open(args_path, 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
            parser.set_defaults(**args)
        args = parser.parse_args()
        LABEL2ID, ID2LABEL = get_label_ID_map(args.num_classes)
        ckpt_path_template = os.path.join(exp_path, 'fold{}_{}_best.pth')
        for i in range(1, 6):
            model, model_config, transform = get_timm_model(args.model_name, num_classes=len(ID2LABEL), pretrained=False, args=args)
            model_path = ckpt_path_template.format(i, args.model_name)
            model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
            model.eval()
            model = model.cuda()
            model_list.append(model)
    
    args.gpu_id = args.gpu = gpu
    args.data_root = 'Validation-Dataset'
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    
    model = EnsembleModel(model_list, ensemble_type)
    model.eval()
    model = model.cuda()
    
    LABEL2ID, ID2LABEL = get_label_ID_map(args.num_classes)
    
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        
    # used for denormalization and visualization
    mean = model_config['mean']
    std = model_config['std']
    mean = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()
    
    # Dataset
    dataset = EsophagealCancerDataset(args=args, transform=transform, image_list=image_path_list, split='test')
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    log_folder = 'validation_output'
    log_path = os.path.join(log_folder, 'ensemble-many_MPR' + '.log')
    prob_log_path = os.path.join(log_folder, 'ensemble-many' + '_prob.csv')
    # save ckpt path
    with open(log_path, 'a') as f:
        for exp_path in exp_path_list:
            f.write(exp_path + '\n')
    
    def test_ensemble(model, dataloader=test_dataloader):        
        pred_list = []
        prob_list = []
        label_list = []
        target_list = []
        wrong_filename_list = []
        acc = 0
        
        for (img, label, filename, patient_id, ood_label) in tqdm(dataloader):

            
            pred, prob = model(img.cuda())
            # print(output_list, pred)
            pred_list.append(pred)
            
            target_list.append(label.item())
            if pred != label:
                wrong_filename_list.append((patient_id.item(), filename[0], ID2LABEL[label.item()], ID2LABEL[pred]))
            else:
                acc += 1
            
            if label == 0:
                label_list.append(1)
                if pred == label:
                    prob_list.append(prob)
                else:
                    prob_list.append(1 - prob)
            else:
                label_list.append(-1)
                if pred == label:
                    prob_list.append(1 - prob)
                else:
                    prob_list.append(prob)
                       
        print(f'accuracy: {acc / len(dataloader)}')
        return pred_list, target_list, prob_list, wrong_filename_list, label_list

    pred_list, target_list, prob_list, wrong_filename_list, label_list = test_ensemble(model)
    
    wrong_filename_list = sorted(wrong_filename_list, key=lambda x: x[0])
    with open(os.path.join(log_folder, 'wrong_list.txt'), 'w') as f:
        f.write(f'patient_id, filename, label, pred\n')
        for patient_id, filename, label, pred in wrong_filename_list:
            f.write(f'{patient_id}, {filename}, {label}, {pred}\n')
        
    # save results
    with open(log_path, 'a') as f:    
        f.write(f'#### ensembling strategy: {model.ensemble_type} ####\n')
        f.write(f'Classification Report:\n{classification_report(target_list, pred_list, target_names=ID2LABEL, digits=4)}\n')
        f.write(f'Confusion Matrix:\n{confusion_matrix(target_list, pred_list)}\n')
        exp_confusion_matrix = confusion_matrix(target_list, pred_list)
        f.write(f'confidence CI:\n')
        f.write(f'{generate_CI(exp_confusion_matrix)}\n')
    # plot roc curve
    display = RocCurveDisplay.from_predictions(
        y_true=np.array(label_list),
        y_pred=np.array(prob_list),
        color="darkorange",
        name="ESCNet",
        plot_chance_level=True,
    )
    _ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    )
    plt.savefig('roc_curve.png')
    display = PrecisionRecallDisplay.from_predictions(
        y_true=np.array(label_list),
        y_pred=np.array(prob_list),
        color="darkorange",
        name="ESCNet",
    )
    _ = display.ax_.set(
    xlabel="Recall",
    ylabel="Precision",
    )
    plt.savefig('precision_recall_curve.png')
    with open(prob_log_path, 'w') as f:
        for prob, label in zip(prob_list, label_list):
            f.write(f'{prob, label}\n')

if __name__ == '__main__':
    import json
    patient_data_info_path = '/mnt/nasv2/hhz/Esophageal-Cancer/Esophageal-Cancer-Dataset/patient_data_has_normal.json'
    with open(patient_data_info_path, 'r') as f:
        patient_data_info = json.load(f)
    exp_path_list = [
        ...
    ]
    
    gpu = 3
    ensemble_type = 'soft'
    

    # args
    parser = argparse.ArgumentParser(description='5-fold cross-validation')
    parser.add_argument('--cfg', type=str, default='cfg/train.yaml', help='config file', metavar='FILE')
    parser.add_argument('--model', type=str, default='test', help='model name', metavar='FILE')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id')
    model = parser.parse_args().model
    gpu = parser.parse_args().gpu
    args, _ = parse_args_and_yaml(parser)
    if model != 'test':
        args.model = model
        args.model_name = model
    if gpu > -1:
        args.gpu = gpu
        args.gpu_id = gpu
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)  
    
    LABEL2ID, ID2LABEL = get_label_ID_map(args.num_classes)
    model, model_config, transform = get_timm_model(args.model_name, num_classes=len(ID2LABEL), pretrained=False, args=args)
    
    def test_ensemble_for_the_first_dataset(model, dataloader):        
        pred_list = []
        prob_list = []
        label_list = []
        target_list = []
        wrong_filename_list = []
        acc = 0
        patient_accurate_count = {}
        for (img, label, filename, patient_id, ood_label) in tqdm(dataloader):
            ####TEMP TODO DELETE
            if 'White-MPR' not in filename[0]:
                continue
            str_id = patient_id.item()
            patient_label = patient_data_info[str(str_id)]['patient_label']
            

            if patient_label != 'MPR':
                continue
            
            pred, prob = model(img.cuda())
            # print(output_list, pred)
            pred_list.append(pred)
            
            target_list.append(label.item())

            if str_id not in patient_accurate_count.keys():
                patient_accurate_count[str_id] = []

            if pred != label:
                wrong_filename_list.append((patient_id.item(), filename[0], ID2LABEL[label.item()], ID2LABEL[pred]))
                patient_accurate_count[str_id].append(0)
            else:
                acc += 1
                patient_accurate_count[str_id].append(1)
            
            if label == 0:
                label_list.append(1)
                if pred == label:
                    prob_list.append(prob)
                else:
                    prob_list.append(1 - prob)
            else:
                label_list.append(-1)
                if pred == label:
                    prob_list.append(1 - prob)
                else:
                    prob_list.append(prob)
                       
        print(f'accuracy: {acc / len(dataloader)}')
        
        wrong_filename_list = sorted(wrong_filename_list, key=lambda x: x[0])
        
        count = len(patient_accurate_count.keys())
        accurate_count = 0
        for patient_id, predictions in patient_accurate_count.items():
            if len(predictions) == 0:
                continue
            if sum(predictions) / len(predictions) > 0.5:
                accurate_count += 1
        print(f'patient accuracy: {accurate_count} / {count} = {accurate_count / count}')

        return pred_list, target_list, prob_list, wrong_filename_list, label_list
    
    log_folder = 'validation_output_for_cross_validation'
    overall_pred_list = []
    overall_target_list = []
    overall_prob_list = []
    overall_label_list = []
    overall_wrong_filename_list = []
    ##TEMP TODO
    args.no_MPR = False
    args.val_data['clean_data'] = True
    
    cross_val_data_solver = FiveCrossDataset(args, transform)
    
    for val_fold_idx in range(1, 6):
        log_path = os.path.join(log_folder, f'fold{val_fold_idx}-ensemble-many_MPR' + '.log')
        prob_log_path = os.path.join(log_folder, f'fold{val_fold_idx}-ensemble-many_MPR' + '_prob.csv')
        
        _, _, test_dataloader = cross_val_data_solver.get_dataloader(val_fold_idx=val_fold_idx)
        # load ensemble model
        model_list = []
        for exp_path in exp_path_list:
            args_path = os.path.join(exp_path, 'args.yaml')
            with open(args_path, 'r', encoding='utf-8') as f:
                args = yaml.safe_load(f)
                parser.set_defaults(**args)
            args = parser.parse_args()
            
            LABEL2ID, ID2LABEL = get_label_ID_map(args.num_classes)
            
            ckpt_path_template = os.path.join(exp_path, 'fold{}_{}_best.pth')
            model, model_config, transform = get_timm_model(args.model_name, num_classes=len(ID2LABEL), pretrained=False, args=args)
            model_path = ckpt_path_template.format(val_fold_idx, args.model_name)
            model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
            model.eval()
            model = model.cuda()
            model_list.append(model)
        model = EnsembleModel(model_list, ensemble_type)
        model.eval()
        model = model.cuda()
        pred_list, target_list, prob_list, wrong_filename_list, label_list = test_ensemble_for_the_first_dataset(model, test_dataloader)
        
        overall_pred_list.extend(pred_list)
        overall_target_list.extend(target_list)
        overall_prob_list.extend(prob_list)
        overall_label_list.extend(label_list)
        overall_wrong_filename_list.extend(wrong_filename_list)
        
        # save results
        with open(log_path, 'a') as f:    
            f.write(f'#### ensembling strategy: {model.ensemble_type} ####\n')
            f.write(f'Classification Report:\n{classification_report(target_list, pred_list, target_names=ID2LABEL, digits=4)}\n')
            f.write(f'Confusion Matrix:\n{confusion_matrix(target_list, pred_list)}\n')
        # plot roc curve
        display = RocCurveDisplay.from_predictions(
            y_true=np.array(label_list),
            y_pred=np.array(prob_list),
            color="darkorange",
            name="ESCNet",
            plot_chance_level=True,
        )
        _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        )
        plt.savefig(os.path.join(log_folder, f'fold{val_fold_idx}-roc_curve.png'))
        with open(prob_log_path, 'w') as f:
            for prob, label in zip(prob_list, label_list):
                f.write(f'{prob, label}\n')
    
    with open(os.path.join(log_folder, 'wrong_list.txt'), 'w') as f:
        f.write(f'patient_id, filename, label, pred\n')
        for patient_id, filename, label, pred in overall_wrong_filename_list:
            f.write(f'{patient_id}, {filename}, {label}, {pred}\n')
    # generate report and plot roc curve
    with open(os.path.join(log_folder, 'ensemble-many' + '.log'), 'a') as f:
        f.write('\n'.join(exp_path_list) + '\n')
        f.write(f'#### ensembling strategy: {ensemble_type} ####\n')
        f.write(f'Classification Report:\n{classification_report(overall_target_list, overall_pred_list, target_names=ID2LABEL, digits=4)}\n')
        f.write(f'Confusion Matrix:\n{confusion_matrix(overall_target_list, overall_pred_list)}\n')
        confusion_matrix = confusion_matrix(overall_target_list, overall_pred_list)
        f.write(f'confidence CI:\n')
        f.write(f'{generate_CI(confusion_matrix)}\n')
    display = RocCurveDisplay.from_predictions(
        y_true=np.array(overall_label_list),
        y_pred=np.array(overall_prob_list),
        color="darkorange",
        name="ESCNet",
        plot_chance_level=True,
    )
    _ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    )
    plt.savefig(os.path.join(log_folder, 'roc_curve.png'))
    
    display = PrecisionRecallDisplay.from_predictions(
        y_true=np.array(overall_label_list),
        y_pred=np.array(overall_prob_list),
        color="darkorange",
        name="ESCNet",
    )
    _ = display.ax_.set(
    xlabel="Recall",
    ylabel="Precision",
    )
    plt.savefig(os.path.join(log_folder, 'precision_recall_curve.png'))
    
    with open(os.path.join(log_folder, 'ensemble-many' + '_prob.csv'), 'w') as f:
        for prob, label in zip(overall_prob_list, overall_label_list):
            f.write(f'{prob, label}\n')
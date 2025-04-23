# 5-fold cross-validation
import argparse
import cv2
import numpy as np
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
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay

from data import FiveCrossDataset, get_label_ID_map, get_hardest_samples
from generate_test_report import generate_cross_validation_report
from model import get_timm_model, get_trained_model, get_local_timm_checkpoint_path
from training import SAM, FocalLoss
from utils import parse_args_and_yaml, set_seed, make_exp_folder, Logger

global LABEL2ID, ID2LABEL
 
def save_model(model, model_name, optimizer, lr_scheduler, epoch, ckpt_path):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else 'None'
    }, ckpt_path)
    print('model saved to', ckpt_path)

@torch.no_grad()
def validate(model, val_dataloader, device, loss_fn):
    print('Start validation')
    model.eval()
    
    average_acc_computer = torchmetrics.Accuracy(task="multiclass", num_classes=len(ID2LABEL)).to(device)
    class_acc_computer = torchmetrics.Accuracy(task="multiclass", num_classes=len(ID2LABEL), average="none").to(device)
    
    total_loss = 0
    pred_list = []
    target_list = []
    labels_title = ID2LABEL
    
    for i, (img, label, filename, patient_id, ood_label) in enumerate(val_dataloader):
        img, label = img.to(device), label.to(device)
        output = model(img)
        loss = loss_fn(output, label)
        total_loss += loss.cpu().item()
        average_acc_computer.update(output, label)
        class_acc_computer.update(output, label)
        
        pred = output.argmax(dim=1)
        pred_list.extend(pred.cpu().tolist())
        target_list.extend(label.cpu().tolist())
    total_loss /= len(val_dataloader)
    average_acc = average_acc_computer.compute().cpu().item()
    class_acc = class_acc_computer.compute().cpu().numpy()
    print('\n', classification_report(target_list, pred_list, target_names=labels_title))
    print('\n', confusion_matrix(target_list, pred_list))
    return total_loss, average_acc, class_acc, pred_list, target_list

@torch.no_grad()
def test(model, test_dataloader, device, loss_fn):
    model.eval()
    
    average_acc_computer = torchmetrics.Accuracy(task="multiclass", num_classes=len(ID2LABEL)).to(device)
    class_acc_computer = torchmetrics.Accuracy(task="multiclass", num_classes=len(ID2LABEL), average="none").to(device)
    
    total_loss = 0
    filename_list = []
    pred_list = []
    prob_list = []
    label_list = []
    target_list = []

    for (img, label, filename, patient_id, ood_label) in tqdm(test_dataloader.dataset):
        img, label = img.unsqueeze(0).to(device), label.unsqueeze(0).to(device)
        output = model(img)
        loss = loss_fn(output, label)
        total_loss += loss.cpu().item()
        average_acc_computer.update(output, label)
        class_acc_computer.update(output, label)
        
        pred = output.argmax(dim=1)
        if label == 0:
            label_list.append(1)
        else:
            label_list.append(-1)
        prob = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
        prob_list.append(prob[0])
            
        filename_list.append(filename)
        pred_list.extend(pred.cpu().tolist())
        target_list.extend(label.cpu().tolist())
        
    total_loss /= len(test_dataloader)
    average_acc = average_acc_computer.compute().cpu().item()
    class_acc = class_acc_computer.compute().cpu().numpy()
    return total_loss, average_acc, class_acc, filename_list, pred_list, target_list, (prob_list, label_list)

def fold_validation_for_one_model(args, model_name, val_fold_idx, device, exp_folder_path):
    if args.imagenet_pretrained:
        ckpt_path = get_local_timm_checkpoint_path(model_name)
        if ckpt_path is not None:
            pretrained = False
        else:
            pretrained = True
    else:
        ckpt_path = None
        pretrained = False
    model, model_config, transform = get_timm_model(model_name, num_classes=len(ID2LABEL), pretrained=pretrained, checkpoint_path=ckpt_path, args=args)
    model = model.to(device)
    
    # create optimizer
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SAM':
        optimizer = SAM(model.parameters(), base_optimizer=SGD, rho=args.rho, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError('optimizer type {} is not supported'.format(args.optimizer))

    # create scheduler
    if args.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0., verbose=True)
    elif args.lr_scheduler == 'none':
        scheduler = None
    else:
        raise ValueError('lr_scheduler type {} is not supported'.format(args.lr_scheduler))
    
    # create loss function
    if args.loss == 'ce':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        loss_fn = FocalLoss()
    else:
        raise ValueError('loss type {} is not supported'.format(args.loss))
    
    # create dataloader
    cross_val_data_solver = FiveCrossDataset(args, transform)
    train_dataloader, val_dataloader, test_dataloader = cross_val_data_solver.get_dataloader(val_fold_idx=val_fold_idx)
    print('len(train_dataloader):', len(train_dataloader))   
    
    # create metric computer
    average_acc_computer = torchmetrics.Accuracy(task="multiclass", num_classes=len(ID2LABEL)).to(device)
    class_acc_computer = torchmetrics.Accuracy(task="multiclass", num_classes=len(ID2LABEL), average="none").to(device)
    
    best_val_average_acc = 0.
    #### train loop starts ####
    print('Start training for fold', val_fold_idx)
    for epoch in range(1, args.epochs+1):
        model.train()
        average_acc_computer.reset()
        class_acc_computer.reset()
        epoch_loss = 0.
        print('epoch', epoch, 'lr', optimizer.param_groups[0]['lr'])

        for i, (img, label, ood_label) in tqdm(enumerate(train_dataloader)):
            img, label = img.to(device), label.to(device)
            output = model(img)

            if args.optimizer == 'SGD' or args.optimizer == 'Adam':
                optimizer.zero_grad()
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()
            elif args.optimizer == 'SAM':
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                loss_fn(model(img), label).backward()
                optimizer.second_step(zero_grad=True)
            else:
                raise ValueError('optimizer type {} is not supported'.format(args.optimizer))
            
            # compute train metrics
            average_acc_computer.update(output, label)
            class_acc_computer.update(output, label)
            epoch_loss += loss.cpu().item()

        if scheduler is not None:
            scheduler.step()
        
        # save latest model
        ckpt_path = os.path.join(exp_folder_path, 'fold{}_{}_latest.pth'.format(val_fold_idx, model_name))
        save_model(model, model_name, optimizer, scheduler, epoch, ckpt_path=ckpt_path)
        
        # compute epoch train metrics
        average_acc = average_acc_computer.compute().cpu().item()
        class_acc = class_acc_computer.compute().cpu().numpy()
        epoch_loss /= len(train_dataloader)
        print('epoch: {}, train_loss: {:.4f}, average_acc: {:.4f}'.format(epoch, epoch_loss, average_acc))
        wandb.log({'train/loss': epoch_loss, 'train/average_acc': average_acc, 'lr': optimizer.param_groups[0]['lr']}, step=epoch+(val_fold_idx-1)*(args.epochs+1))
        for i in range(len(ID2LABEL)):
            print('{}: {:.4f}'.format(ID2LABEL[i], class_acc[i]))
            wandb.log({f'train/{ID2LABEL[i]}': class_acc[i]}, step=epoch+(val_fold_idx-1)*(args.epochs+1))
        
        # validate
        if epoch % args.val_interval == 0:
            val_loss, val_average_acc, val_class_acc, val_pred_list, val_target_list = validate(model, val_dataloader, device, loss_fn)
            print('val_loss: {:.4f}, val_average_acc: {:.4f}'.format(val_loss, val_average_acc))
            if val_average_acc > best_val_average_acc:
                best_val_average_acc = val_average_acc
                ckpt_path = os.path.join(exp_folder_path, 'fold{}_{}_best.pth'.format(val_fold_idx, model_name))
                save_model(model, model_name, optimizer, scheduler, epoch, ckpt_path=ckpt_path)
            wandb.log({'val/loss': val_loss, 'val/average_acc': val_average_acc}, step=epoch+(val_fold_idx-1)*(args.epochs+1))
            for i in range(len(ID2LABEL)):
                print('{}: {:.4f}'.format(ID2LABEL[i], val_class_acc[i]))
                wandb.log({f'val/{ID2LABEL[i]}': val_class_acc[i]}, step=epoch+(val_fold_idx-1)*(args.epochs+1))
    #### train loop ends ####
    
    # test
    if args.epochs == 0:
        epoch = 0
    print('Start testing for fold', val_fold_idx)
    ckpt_path = os.path.join(exp_folder_path, 'fold{}_{}_best.pth'.format(val_fold_idx, model_name))
    model = get_trained_model(model_name, len(ID2LABEL), ckpt_path)
    model = model.to(device)
    test_loss, test_average_acc, test_class_acc, test_filename_list, test_pred_list, test_target_list, roc_plot_data = test(model, test_dataloader, device, loss_fn)
    print('test_loss: {:.4f}, test_average_acc: {:.4f}'.format(test_loss, test_average_acc))
    wandb.log({f'test/loss': test_loss, f'test/average_acc': test_average_acc}, step=epoch+(val_fold_idx-1)*(args.epochs+1))
    for i in range(len(ID2LABEL)):
        print('{}: {:.4f}'.format(ID2LABEL[i], test_class_acc[i]))
        wandb.log({f'test/{ID2LABEL[i]}': test_class_acc[i]}, step=epoch+(val_fold_idx-1)*(args.epochs+1))
    print('\n', classification_report(test_target_list, test_pred_list, target_names=ID2LABEL))
    print('\n', confusion_matrix(test_target_list, test_pred_list))
    # save test result to csv for further statistical analysis
    with open(os.path.join(exp_folder_path, 'fold{}_test_result.csv'.format(val_fold_idx)), 'w') as f:
        f.write('filename,pred,target\n')
        for filename, pred, target in zip(test_filename_list, test_pred_list, test_target_list):
            f.write('{},{},{}\n'.format(filename, pred, target))
    # plot roc curve
    display = RocCurveDisplay.from_predictions(
        y_true=np.array(roc_plot_data[1]),
        y_pred=np.array(roc_plot_data[0]),
        color="darkorange",
        name="{}".format(model_name.split('.')[0].split('_')[0]),
        plot_chance_level=True,
    )
    _ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    )
    plt.savefig(f'{exp_folder_path}/fold{val_fold_idx}_roc_curve.png')
    # save roc plot data
    with open(os.path.join(exp_folder_path, 'fold{}_roc_plot_data.csv'.format(val_fold_idx)), 'w') as f:
        f.write('prob,label\n')
        for prob, label in zip(roc_plot_data[0], roc_plot_data[1]):
            f.write('{},{}\n'.format(prob, label))
    print('test result saved to', os.path.join(exp_folder_path, 'fold{}_test_result.csv'.format(val_fold_idx)))
    
    
def cross_validation_for_one_model(args, model_name, device, exp_folder_path):
    if 'ception' in model_name:
        args.input_size = 299
    else:
        args.input_size = 224
    for val_fold_idx in range(1, 6):
        fold_validation_for_one_model(args, model_name, val_fold_idx, device, exp_folder_path)
    raise ValueError('temp')
    generate_cross_validation_report(exp_folder_path)

def main(args, exp_folder_path, model_name):
    set_seed(args.seed)

    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        if torch.cuda.is_available():
            device = torch.device('cuda') 
        else:
            raise ValueError('GPU is not available')
        print('Using GPU, device: ', device)
    else:
        device = torch.device('cpu')
        print('Using CPU')
        
    cross_validation_for_one_model(args, model_name, device, exp_folder_path)
    
    print('finish evaluation for models: ', model_name)
    
if __name__ == '__main__':
    # add environment variable for huggingface mirror
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print('HF_ENDPOINT:', os.environ['HF_ENDPOINT'])

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
    
    # global variables
    LABEL2ID, ID2LABEL = get_label_ID_map(args.num_classes)
    
    if args.model_name is not None:
        if 'ception' in args.model_name:
            args.input_size = 229
        exp_folder_path = make_exp_folder(args.exp_name + '_{}'.format(args.model_name.split('.')[0]))
        sys.stdout = Logger(os.path.join(exp_folder_path, 'log.txt'))
        print(args_text)
        # save args to yaml file under exp_folder_path
        with open(os.path.join(exp_folder_path, 'args.yaml'), 'w') as f:
            yaml.dump(args.__dict__, f)
        # save hard sampels if needed
        if args.clean_data and args.example_exp_folder is not None:
            _, hardest_samples_img_path_list = get_hardest_samples(args.example_exp_folder)
            with open(os.path.join(exp_folder_path, 'hard_samples.txt'), 'w') as f:
                for img_path in hardest_samples_img_path_list:
                    f.write(img_path+'\n')

        wandb.init(project="Esophageal-Cancer", config=args.__dict__, name=args.exp_name+ '_{}'.format(args.model_name.split('.')[0]),
               tags=['class'+str(args.num_classes), '-'.join(args.imaging_type)])
        main(args, exp_folder_path, args.model_name)
    else:
        raise ValueError('model_name is None')
    
    
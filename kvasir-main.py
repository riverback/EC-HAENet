import argparse
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from scipy.stats import beta
import yaml

from kvasir import KvasirDataset, get_label_ID_map
from model import get_timm_model, get_trained_model, get_local_timm_checkpoint_path
from utils import parse_args_and_yaml, set_seed, Logger
from confidence_interval import generate_CI
from training import FocalLoss, SAM

ID2LABEL, LABEL2ID = get_label_ID_map()

def make_exp_folder(exp_name):
    """
    Make a folder for the experiment, and return the path,
    if the folder already exists, ask the user to confirm whether to overwrite it.
    """
    exp_folder = os.path.join('kvasir-checkpoints', exp_name)
    if os.path.exists(exp_folder):
        print('The folder {} already exists, do you want to overwrite it?'.format(exp_folder))
        while True:
            choice = input('Please input y/n: ')
            if choice == 'y':
                break
            elif choice == 'n':
                print('Please change the exp_name in the config file')
                sys.exit(0)
            else:
                print('Invalid input, please input y/n')
    else:
        os.makedirs(exp_folder)
    return exp_folder

def save_model(model, model_name, optimizer, lr_scheduler, epoch, ckpt_path):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else 'None'
    }, ckpt_path)
    print('model saved to', ckpt_path)

def clopper_pearson_interval(number_of_successes, number_of_trials, alpha=0.05):
    p_u, p_o = beta.ppf([alpha/2, 1-alpha/2], 
        [number_of_successes, number_of_successes + 1],
        [number_of_trials - number_of_successes + 1, number_of_trials - number_of_successes])

    #print(f'{number_of_successes / number_of_trials:.4f}', f'({p_u:.4f}-{p_o:.4f})')
    return f'{number_of_successes / number_of_trials:.4f} ({p_u:.4f}-{p_o:.4f})\n'

@torch.no_grad()
def validate(model, val_loader, device, loss_fn):
    model.eval()
    
    average_acc_computer = torchmetrics.Accuracy(task="multiclass", num_classes=len(ID2LABEL)).to(device)
    class_acc_computer = torchmetrics.Accuracy(task="multiclass", num_classes=len(ID2LABEL), average="none").to(device)
    
    total_loss = 0
    pred_list = []
    target_list = []
    labels_title = ID2LABEL
    
    for i, (img, label) in enumerate(val_loader):
        img, label = img.to(device), label.to(device)
        output = model(img)
        loss = loss_fn(output, label)
        total_loss += loss.cpu().item()
        average_acc_computer.update(output, label)
        class_acc_computer.update(output, label)
        
        pred = output.argmax(dim=1)
        pred_list.extend(pred.cpu().tolist())
        target_list.extend(label.cpu().tolist())

    average_acc = average_acc_computer.compute().cpu().item()
    class_acc = class_acc_computer.compute().cpu().numpy()
    print('\n', classification_report(target_list, pred_list, target_names=labels_title))
    print('\n', confusion_matrix(target_list, pred_list))
    return total_loss, average_acc, class_acc, pred_list, target_list

@torch.no_grad()
def test(model, test_loader, device, loss_fn):

    model.eval()
    average_acc_computer = torchmetrics.Accuracy(task="multiclass", num_classes=len(ID2LABEL)).to(device)
    class_acc_computer = torchmetrics.Accuracy(task="multiclass", num_classes=len(ID2LABEL), average="none").to(device)
    
    total_loss = 0
    pred_list = []
    prob_list = []
    target_list = []

    for i, (img, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
        img, label = img.to(device), label.to(device)
        output = model(img)
        prob = torch.nn.functional.softmax(output, dim=1)
        loss = loss_fn(output, label)
        total_loss += loss.cpu().item()
        average_acc_computer.update(output, label)
        class_acc_computer.update(output, label)
        
        pred = output.argmax(dim=1)
        pred_list.extend(pred.cpu().tolist())
        target_list.extend(label.cpu().tolist())
        prob_list.extend(prob.cpu().tolist())

    average_acc = average_acc_computer.compute().cpu().item()
    class_acc = class_acc_computer.compute().cpu().numpy()
    return total_loss, average_acc, class_acc, pred_list, target_list, prob_list

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
    if args.imagenet_pretrained:
        ckpt_path = get_local_timm_checkpoint_path(model_name)
        if ckpt_path is not None:
            pretrained = False
        else:
            pretrained = True
    else:
        ckpt_path = None
        pretrained = False
    model, model_config, transform = get_timm_model(model_name=model_name, num_classes=len(ID2LABEL), 
                        checkpoint_path=ckpt_path, pretrained=True, args=args)
    model = model.to(device)
    # create optimizer
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Invalid optimizer')
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
    
    train_dataset = KvasirDataset(args, 'train', transform)
    train_loader = DataLoader(train_dataset, batch_size=args.train_data['batch_size'], shuffle=True, num_workers=args.train_data['num_workers'])
    val_dataset = KvasirDataset(args, 'val', transform)
    val_loader = DataLoader(val_dataset, batch_size=args.train_data['batch_size'], shuffle=False, num_workers=args.train_data['num_workers'])
    test_dataset = KvasirDataset(args, 'test', transform)
    test_loader = DataLoader(test_dataset, batch_size=args.val_data['batch_size'], shuffle=False, num_workers=args.val_data['num_workers'])
    
    # create metrics computer
    average_acc_computer = torchmetrics.Accuracy(task="multiclass", num_classes=len(ID2LABEL)).to(device)
    class_acc_computer = torchmetrics.Accuracy(task="multiclass", num_classes=len(ID2LABEL), average="none").to(device)

    best_val_average_acc = 0.
    print('start training')
    for epoch in range(1, args.epochs+1):
        print('epoch:', epoch)
        model.train()
        average_acc_computer.reset()
        class_acc_computer.reset()
        epoch_loss = 0.

        for i, (img, label) in tqdm(enumerate(train_loader), total=len(train_loader)//args.train_data['batch_size']):
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.cpu().item()
            average_acc_computer.update(output, label)
            class_acc_computer.update(output, label)
        if scheduler is not None:
            scheduler.step()

        # save latest model
        ckpt_path = os.path.join(exp_folder_path, '{}_latest.pth'.format(model_name))
        save_model(model, model_name, optimizer, scheduler, epoch, ckpt_path=ckpt_path)
        
        # compute epoch train metrics
        average_acc = average_acc_computer.compute().cpu().item()
        class_acc = class_acc_computer.compute().cpu().numpy()
        epoch_loss /= len(train_loader)
        print('epoch: {}, train_loss: {:.4f}, average_acc: {:.4f}'.format(epoch, epoch_loss, average_acc))
        wandb.log({'train/loss': epoch_loss, 'train/average_acc': average_acc, 'lr': optimizer.param_groups[0]['lr']}, step=epoch)
        for i in range(len(ID2LABEL)):
            print('{}: {:.4f}'.format(ID2LABEL[i], class_acc[i]))
            wandb.log({f'train/{ID2LABEL[i]}': class_acc[i]}, step=epoch)

        # validation
        if epoch % args.val_interval == 0:
            val_loss, val_average_acc, val_class_acc, val_pred_list, val_target_list = validate(model, val_loader, device, loss_fn)
            print('val_loss: {:.4f}, val_average_acc: {:.4f}'.format(val_loss, val_average_acc))
            if val_average_acc > best_val_average_acc:
                best_val_average_acc = val_average_acc
                ckpt_path = os.path.join(exp_folder_path, '{}_best.pth'.format(model_name))
                save_model(model, model_name, optimizer, scheduler, epoch, ckpt_path=ckpt_path)
            wandb.log({'val/loss': val_loss, 'val/average_acc': val_average_acc}, step=epoch)
            for i in range(len(ID2LABEL)):
                print('{}: {:.4f}'.format(ID2LABEL[i], val_class_acc[i]))
                wandb.log({f'val/{ID2LABEL[i]}': val_class_acc[i]}, step=epoch)
    #### train loop ends ####
    
    # test
    ckpt_path = os.path.join(exp_folder_path, '{}_best.pth'.format(model_name))
    model = get_trained_model(model_name, num_classes=len(ID2LABEL), checkpoint_path=ckpt_path)
    model = model.to(device)
    test_loss, test_average_acc, test_class_acc, test_pred_list, test_target_list, test_prob_list = test(model, test_loader, device, loss_fn)
    test_confusion_matrix = confusion_matrix(test_target_list, test_pred_list)
    number_of_successes = np.trace(test_confusion_matrix)
    number_of_trials = np.sum(test_confusion_matrix)
    print('test_loss: {:.4f}, test_average_acc: {:.4f}'.format(test_loss, test_average_acc))
    print('confusion_matrix:\n', test_confusion_matrix)
    print('accuracy', clopper_pearson_interval(number_of_successes, number_of_trials))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='5-fold cross-validation')
    parser.add_argument('--cfg', type=str, default='cfg/kvasir.yaml', help='config file', metavar='FILE')
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
    
    if args.model_name is not None:
        if 'ception' in args.model_name:
            args.input_size = 229
        exp_folder_path = make_exp_folder(args.exp_name + '{}_{}'.format(args.data_version, args.model_name.split('.')[0]))
        sys.stdout = Logger(os.path.join(exp_folder_path, 'log.txt'))
        print(args_text)
        # save args to yaml file under exp_folder_path
        with open(os.path.join(exp_folder_path, 'args.yaml'), 'w') as f:
            yaml.dump(args.__dict__, f)

        wandb.init(project="Kvasir-classify", config=args.__dict__, name=args.exp_name+ '_{}'.format(args.model_name.split('.')[0]),
               tags=[args.data_version, args.model_name.split('.')[0], args.optimizer, args.loss])
        main(args, exp_folder_path, args.model_name)
    else:
        raise ValueError('model_name is None')
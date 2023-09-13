from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import argparse
import os
import shutil
import time
from datetime import date
from PIL import Image
import random
import copy
import contextlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp

import models.wideresnet as wrn_models
import models.resnet as res_models
import torchvision

#import load_data.datasets as dataset
from load_data.ssl_dataset import SSL_Dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p
from loss import *


parser = argparse.ArgumentParser(description='PyTorch Adversarially Robust Training in SemiSuperversed Learning')

########################## model setting ##########################
parser.add_argument('--depth', type=int, default=28, help='wideresnet depth factor')
parser.add_argument('--widen_factor', type=int, default=5, help='wideresnet widen factor')
parser.add_argument('--activation', type=str, default= 'relu', choices=['relu', 'leaky', 'silu'], help='choice of activation')
parser.add_argument('--model', type=str, default= 'wideresnet', help='architecture of model') #, choices=['resnet18, wideresnet'] : invalid choice

########################## optimization setting ##########################
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restayts)')
parser.add_argument('--lb_batch_size', default=64, type=int, metavar='N', help='train batchsize')
parser.add_argument('--ul_batch_size', default=128, type=int, metavar='N', help='train batchsize')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--wd', default=5e-4, type=float, metavar='WD', help='weight decay')
parser.add_argument('--lr_scheduler', type=str, default= 'Cosine', choices=['MultiStep', 'Cosine', 'Cyclic'], help='learning rate scheduling')
#parser.add_argument('--eval_freq', default=5, type=int, metavar='N', help='frequency of evaluation')
parser.add_argument('--amp', default=True, action='store_false', help='use of amp')

######################### Checkpoints #############################
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

########################## basic setting ##########################
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--gpu', default=0, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--data_dir', default='/home/ydy0415/data/datasets', help='Directory of dataset')
parser.add_argument('--tolerance', default=150, type=int, metavar='N', help='tolerance')

######################### Dataset #############################
parser.add_argument('--dataset', type=str, default= 'cifar10', choices=['cifar10', 'cifar100', 'stl10', 'svhn', 'tiny-imagenet'], help='benchmark dataset')
parser.add_argument('--num_labels', type=int, default= 4000, help='number of labels in semi-supervised learning')
parser.add_argument('--num_classes', type=int, default= 10, help='number of class of dataset')
parser.add_argument('--save_dir', type=str, default='/home/ydy0415/data/experiments/semi-arow', help='save directory')

########################## loss setting ##########################
parser.add_argument('--algo', metavar='ALGO', default='fixmatch', choices=['fixmatch', 'sup', 'kd-semi', 'kd-sup'], help='surrogate loss function to optimize')
parser.add_argument('--lamb', type=float, default=1, help='coefficient of regularization loss')
parser.add_argument('--tau', type=float, default=1, help='coefficient of regularization loss')
parser.add_argument('--eta', type=float, default=0.95, help='threshold of selecting samples')
parser.add_argument('--teacher', type=str, default='fixmatch')

######################### add name #############################
parser.add_argument('--add_name', default='', type=str, help='add_name')


args = parser.parse_args()
print(args)

state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.cuda.set_device(args.gpu)
use_cuda = torch.cuda.is_available()


# Random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic=True

# To speed up training
torch.backends.cudnn.benchmark = True


if args.dataset in ['cifar10', 'cifar100', 'stl', 'svhn']:
    input_channel = 3
elif args.dataset in ['fmnist']:
    input_channel = 1

if args.dataset in ['tiny-imagenet']:
    args.data_dir = os.path.join(args.data_dir, 'tiny-imagenet-200')
    import models.ti_resnet as res_models
    
best_acc = 0  # best val accuracy
tolerance = 0 # tolerance

test_acc = 0

def main(args):
    global best_acc
    global tolerance
    
    global test_acc
    
    args.save_dir += f'/{args.dataset}/'
    args.save_dir += str(date.today().strftime('%Y%m%d')[2:])
    
    out_directory = f'{args.save_dir}/'
    
    if args.model == "wideresnet":
        out_directory += f'wrn-{args.depth}-{args.widen_factor}'
        
    elif args.model == "resnet":
        out_directory += f'rn-{args.depth}'
    
    out_directory += f'_algo-{args.algo}_lrsche-{args.lr_scheduler}'
    
    
    out_directory += f'_numlabels-{args.num_labels}'
    
    if args.algo in ['kd-semi', 'kd-sup']:
        out_directory += f'_tau-{args.tau}_lamb-{args.lamb}'
    
    # first penalty   
    
    out_directory += f'_seed-{args.seed}'
        
    if args.add_name != '':
        out_directory += f'_{args.add_name}'
    
    if not os.path.isdir(out_directory):
        mkdir_p(out_directory)
    
    # Data
    print('==> Preparing ' + str(args.dataset))
    
    data_directory = args.data_dir
    
    train_dataset = SSL_Dataset(args, name=args.dataset, train=True, num_classes=args.num_classes, data_dir=args.data_dir)
    
    lb_dataset, ulb_dataset = train_dataset.get_ssl_dset(num_labels=args.num_labels, index=None, include_lb_to_ulb=True, strong_transform=True, onehot=False)
    
    if args.dataset in ['tiny-imagenet']:
        lb_img = [np.array(Image.open(path).convert("RGB")) for _, (path, _)  in enumerate(lb_dataset.data)]
        lb_dataset.data = np.array(lb_img)
        
        ulb_img = [np.array(Image.open(path).convert("RGB")) for _, (path, _)  in enumerate(ulb_dataset.data)]
        ulb_dataset.data = np.array(ulb_img)
        
        del lb_img, ulb_img
        #gc.collect()
    
    
    if args.algo not in ['fixmatch']:
        labeled_trainloader = torch.utils.data.DataLoader(lb_dataset, batch_size=args.lb_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        unlabeled_trainloader = torch.utils.data.DataLoader(ulb_dataset, batch_size=args.ul_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    else:
        labeled_trainloader = torch.utils.data.DataLoader(lb_dataset, batch_size=args.lb_batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        unlabeled_trainloader = torch.utils.data.DataLoader(ulb_dataset, batch_size=args.ul_batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    if args.algo in ["kd-semi"]:
        teacher_model = wrn_models.WideResNet(first_stride = 2 if args.dataset in ['stl10'] else 1,
                                          num_classes  = args.num_classes,
                                          depth        = args.depth,
                                          widen_factor = args.widen_factor,
                                          activation   = args.activation).cuda(args.gpu)
    
    
        checkpoint = torch.load(f'/home/ydy0415/data/teacher/{args.dataset}/wrn-{args.depth}-{args.widen_factor}_teacher-{args.teacher}_numlabels-{4000}.pth.tar'
                                , map_location='cuda:' + str(args.gpu))
        #teacher_model.load_state_dict(checkpoint['ema_state_dict'])
        teacher_model.load_state_dict(checkpoint['state_dict'])
        teacher_model.eval()
        del checkpoint
        torch.cuda.empty_cache()
        print("==> The teacher-model is loaded.")
    
    if args.algo in ["kd-sup"]:
        teacher_model = wrn_models.WideResNet(first_stride = 2 if args.dataset in ['stl10'] else 1,
                                          num_classes  = args.num_classes,
                                          depth        = args.depth,
                                          widen_factor = args.widen_factor,
                                          activation   = args.activation).cuda(args.gpu)
    
    
        checkpoint = torch.load(f'/home/ydy0415/data/teacher/{args.dataset}/wrn-{args.depth}-{args.widen_factor}_teacher-{args.teacher}_numlabels-{50000}.pth.tar'
                                , map_location='cuda:' + str(args.gpu))
        #teacher_model.load_state_dict(checkpoint['ema_state_dict'])
        teacher_model.load_state_dict(checkpoint['state_dict'])
        teacher_model.eval()
        del checkpoint
        torch.cuda.empty_cache()
        print("==> The teacher-model is loaded.")
    
    test_transform = transforms.Compose([transforms.ToTensor()])
    
    if args.dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)
    elif args.dataset == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=test_transform)
    elif args.dataset == 'svhn':
        test_dataset = torchvision.datasets.SVHN(root=args.data_dir, split= 'test', download=True, transform=test_transform) 
    elif args.dataset == 'stl10':
        test_dataset = torchvision.datasets.STL10(root=args.data_dir, split= 'test', download=True, transform=test_transform)
    elif args.dataset == 'tiny-imagenet':
        test_dataset = ImageFolder(os.path.join(args.data_dir, 'val'), transform=test_transform)
        
        def get_annotations_map(path):
            valAnnotationsPath = os.path.join(path, 'val/val_annotations.txt')
            valAnnotationsFile = open(valAnnotationsPath, 'r')
            valAnnotationsContents = valAnnotationsFile.read()
            valAnnotations = {}

            for line in valAnnotationsContents.splitlines():
                pieces = line.strip().split()
                valAnnotations[pieces[0]] = pieces[1]

            return valAnnotations
        
        file_to_class=get_annotations_map(args.data_dir)
        
        for key, value in file_to_class.items():
            file_to_class[key] = class_to_idx.get(value, value)
        
        test_dataset.targets= list(file_to_class.values())
        
        
    else:
        raise ValueError("dataset error.")
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.ul_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    def create_model(args):
        if args.model == 'wideresnet':
            print("==> creating WideResNet" + str(args.depth) + '-' + str(args.widen_factor))
            model = wrn_models.WideResNet(first_stride = 2 if args.dataset in ['stl10'] else 1,
                                              num_classes  = args.num_classes,
                                              depth        = args.depth,
                                              widen_factor = args.widen_factor,
                                              activation   = args.activation).cuda(args.gpu)

            return model
        
        elif args.model == 'resnet':
            print("==> creating ResNet" + str(args.depth))
            model = res_models.resnet('resnet' + str(args.depth), input_channel, num_classes=args.num_classes).cuda(args.gpu)
            
            return model
    
    model = create_model(args)
        
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    
    if args.lr_scheduler == "MultiStep":
        scheduler = lr_scheduler.MultiStepLR(optimizer , milestones=[60, 90], gamma=0.1)
            
    elif args.lr_scheduler == "Cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    title = args.dataset
    logger = Logger(os.path.join(out_directory, 'log.txt'), title=title)
    logger.set_names(['Epoch', 'Train Loss', 'Test Loss', 'Test Acc.'])

    # Train and val
    for epoch in range(args.start_epoch, args.epochs + 1):

        print('\n' + ' Epoch: [%d | %d] LR: %.5f Tol: %d Best ts acc: %.2f ' % (epoch, args.epochs, optimizer.param_groups[0]['lr'], tolerance, best_acc))
        if args.algo in ['fixmatch']:
            train_loss = train_semi(args, labeled_trainloader, unlabeled_trainloader, epoch, model, optimizer, use_cuda)
                    #def train_semi(args, labeled_trainloader, unlabeled_trainloader, epoch, model, optimizer, use_cuda):
        elif args.algo in ['sup']:
            train_loss = train_sup(args, labeled_trainloader, epoch, model, optimizer, use_cuda)
                    #def train_sup(args, labeled_trainloader, epoch, model, optimizer, use_cuda):
        elif args.algo in ['kd-semi']:
            train_loss = train_kd_semi(args, labeled_trainloader, unlabeled_trainloader, epoch, model, optimizer, use_cuda, teacher_model)
                    #def train_sup(args, labeled_trainloader, epoch, model, optimizer, use_cuda):
        elif args.algo in ['kd-sup']:
            train_loss = train_kd_sup(args, labeled_trainloader, epoch, model, optimizer, use_cuda, teacher_model)
                    #def train_sup(args, labeled_trainloader, epoch, model, optimizer, use_cuda):
        
        if epoch == 1 or epoch % 5 == 0 or epoch >= args.epochs - 20 :
            test_loss, test_acc = validate(test_loader, model, criterion, use_cuda, mode='Test')
                             #def validate(val_loader, model, criterion, use_cuda, mode, attack=None):
            logger.append([round(epoch), train_loss, test_loss, test_acc])

        if args.lr_scheduler in ["MultiStep", "Cosine"]:
            scheduler.step()
        elif args.lr_scheduler == "Cyclic":
            if epoch < 30:
                pass
            elif (epoch >= 30) & (epoch < 50):
                scheduler1.step()
            elif epoch >= 50:
                scheduler2.step()
        
        # save model
        
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        
        if is_best:
            best_acc = test_acc
            save_checkpoint(out_directory, epoch, 
                filename='best.pth.tar',       
                state_dict = model.state_dict(),
                test_acc = test_acc,
                optimizer = optimizer.state_dict()
                )
            
        if epoch == args.epochs:
            save_checkpoint(out_directory, epoch, 
            filename='last.pth.tar',
            state_dict = model.state_dict(),
            test_acc = test_acc
            )


        if is_best:
            tolerance = 0
        else:
            tolerance += 1

    logger.close()

    print('Best test acc:')
    print(best_acc)

def train_semi(args, labeled_trainloader, unlabeled_trainloader, epoch, model, optimizer, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    sup_losses = AverageMeter()
    reg_losses = AverageMeter()
    losses = AverageMeter()
    
    ce_loss=nn.CrossEntropyLoss()
    end = time.time()
    
    scaler = amp.GradScaler()
    amp_cm = amp.autocast if args.amp else contextlib.nullcontext
        
    labeled_train_iter = iter(labeled_trainloader)
    bar = Bar('{:>12}'.format('Training'), max=len(unlabeled_trainloader))
    
    model.train()
    
    for batch_idx, (_, inputs_ul_w, inputs_ul_s) in enumerate(unlabeled_trainloader):

        data_time.update(time.time() - end)
        batch_size = inputs_ul_w.size(0)
        
        try:
            _, inputs_lb, targets_lb = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            _, inputs_lb, targets_lb = next(labeled_train_iter)
        
        if use_cuda:
            inputs_lb, targets_lb, inputs_ul_w, inputs_ul_s = inputs_lb.cuda(), targets_lb.cuda(non_blocking=True), inputs_ul_w.cuda(), inputs_ul_s.cuda()
                
        with amp_cm():
            
            if args.algo == "fixmatch":
                sup_loss, reg_loss = Fix_Match_loss(inputs_lb, targets_lb, inputs_ul_w, inputs_ul_s, model, args.eta)
                                #def Fix_Match_loss(inputs_lb, targets_lb, inputs_ul_w, inputs_ul_s, model, eta): 
                reg_loss = args.lamb * reg_loss
                loss = sup_loss + reg_loss
            
        # record loss
        
        sup_losses.update(sup_loss.item(), inputs_lb.size(0))
        reg_losses.update(reg_loss, inputs_ul_w.size(0))
        losses.update(loss.item(), inputs_ul_w.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch:>3}/{size:>3}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Sup_loss: {sup_loss:.4f} | Reg_loss: {reg_loss:.4f} | Tot loss:{loss:.4f}'.format(
                    batch   = batch_idx + 1,
                    size    = len(unlabeled_trainloader),
                    data    = data_time.avg,
                    bt      = batch_time.avg,
                    total   = bar.elapsed_td,
                    eta     = bar.eta_td,
                    sup_loss=sup_losses.avg,
                    reg_loss=reg_losses.avg,
                    loss=losses.avg
                    )
        bar.next()
    bar.finish()
                  
    return losses.avg

def train_sup(args, labeled_trainloader, epoch, model, optimizer, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    sup_losses = AverageMeter()
    losses = AverageMeter()
    
    ce_loss=nn.CrossEntropyLoss()
    end = time.time()
    
    scaler = amp.GradScaler()
    amp_cm = amp.autocast if args.amp else contextlib.nullcontext
        
    bar = Bar('{:>12}'.format('Training'), max=len(labeled_trainloader))
    
    model.train()
    
    for batch_idx, (_, inputs, targets) in enumerate(labeled_trainloader):

        data_time.update(time.time() - end)
        batch_size = inputs.size(0)
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
                
        with amp_cm():
            loss = ce_loss(model(inputs), targets)
        # record loss
        
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch:>3}/{size:>3}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Tot loss:{loss:.4f}'.format(
                    batch   = batch_idx + 1,
                    size    = len(labeled_trainloader),
                    data    = data_time.avg,
                    bt      = batch_time.avg,
                    total   = bar.elapsed_td,
                    eta     = bar.eta_td,
                    loss    = losses.avg
                    )
        bar.next()
    bar.finish()
                  
    return losses.avg

def train_kd_semi(args, labeled_trainloader, unlabeled_trainloader, epoch, model, optimizer, use_cuda, teacher_model):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    sup_losses = AverageMeter()
    reg_losses = AverageMeter()
    losses = AverageMeter()
    
    ce_loss=nn.CrossEntropyLoss()
    end = time.time()
    
    scaler = amp.GradScaler()
    amp_cm = amp.autocast if args.amp else contextlib.nullcontext
        
    labeled_train_iter = iter(labeled_trainloader)
    bar = Bar('{:>12}'.format('Training'), max=len(unlabeled_trainloader))
    
    model.train()
    
    for batch_idx, (_, inputs_ul_w, inputs_ul_s) in enumerate(unlabeled_trainloader):

        data_time.update(time.time() - end)
        batch_size = inputs_ul_w.size(0)
        
        try:
            _, inputs_lb, targets_lb = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            _, inputs_lb, targets_lb = next(labeled_train_iter)
        
        if use_cuda:
            inputs_lb, targets_lb, inputs_ul_w, inputs_ul_s = inputs_lb.cuda(), targets_lb.cuda(non_blocking=True), inputs_ul_w.cuda(), inputs_ul_s.cuda()
                
        with amp_cm():
            
            if args.algo == "kd-semi":
                sup_loss, reg_loss = KD_loss(inputs_lb,
                                              targets_lb,
                                              inputs_ul_w,
                                              model,
                                              args.tau,
                                              teacher_model)
                reg_loss = args.lamb * reg_loss
                loss = 0.1* sup_loss + 0.9 * (args.tau)**2 *reg_loss
            
        # record loss
        
        sup_losses.update(sup_loss.item(), inputs_lb.size(0))
        reg_losses.update(reg_loss, inputs_ul_w.size(0))
        losses.update(loss.item(), inputs_ul_w.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch:>3}/{size:>3}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Sup_loss: {sup_loss:.4f} | Reg_loss: {reg_loss:.4f} | Tot loss:{loss:.4f}'.format(
                    batch   = batch_idx + 1,
                    size    = len(unlabeled_trainloader),
                    data    = data_time.avg,
                    bt      = batch_time.avg,
                    total   = bar.elapsed_td,
                    eta     = bar.eta_td,
                    sup_loss=sup_losses.avg,
                    reg_loss=reg_losses.avg,
                    loss=losses.avg
                    )
        bar.next()
    bar.finish()
                  
    return losses.avg


def train_kd_sup(args, labeled_trainloader, epoch, model, optimizer, use_cuda, teacher_model):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    sup_losses = AverageMeter()
    losses = AverageMeter()
    
    ce_loss=nn.CrossEntropyLoss()
    end = time.time()
    
    scaler = amp.GradScaler()
    amp_cm = amp.autocast if args.amp else contextlib.nullcontext
        
    bar = Bar('{:>12}'.format('Training'), max=len(labeled_trainloader))
    
    model.train()
    
    for batch_idx, (_, inputs, targets) in enumerate(labeled_trainloader):

        data_time.update(time.time() - end)
        batch_size = inputs.size(0)
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
                
        with amp_cm():
            if args.algo == "kd-sup":
                sup_loss, reg_loss = KD_loss(inputs,
                                             targets,
                                             inputs,
                                             model,
                                             args.tau,
                                             teacher_model)
                reg_loss = args.lamb * reg_loss
                loss = 0.1* sup_loss + 0.9 * (args.tau)**2 *reg_loss
        # record loss
        
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch:>3}/{size:>3}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Tot loss:{loss:.4f}'.format(
                    batch   = batch_idx + 1,
                    size    = len(labeled_trainloader),
                    data    = data_time.avg,
                    bt      = batch_time.avg,
                    total   = bar.elapsed_td,
                    eta     = bar.eta_td,
                    loss    = losses.avg
                    )
        bar.next()
    bar.finish()
                  
    return losses.avg

def validate(val_loader, model, criterion, use_cuda, mode, attack=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    bar = Bar('{mode:>12}'.format(mode=mode), max=len(val_loader))
   
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        
        # compute output
        if attack is not None:
            adv_inputs, _ = attack.perturb(inputs, targets)
            outputs = model(adv_inputs)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch:>3}/{size:>3}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
        bar.next()
    bar.finish()
        
    return (losses.avg, top1.avg)

def save_checkpoint(out_dir, epoch, filename='checkpoint.pth.tar', **kwargs):
    state={
        'epoch' : epoch
    }
    state.update(kwargs)
    filepath = os.path.join(out_dir, filename)
    torch.save(state, filepath)
    
    print("==> saving best model")



if __name__ == '__main__':
    main(args)
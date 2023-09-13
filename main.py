from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import argparse
import os
import shutil
import time
from datetime import date
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
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp

import models.wideresnet as wrn_models
import models.resnet as res_models
import torchvision

#import load_data.datasets as dataset
from load_data.ssl_dataset import SSL_Dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p
from utils.swa import moving_average, bn_update
from loss import *
from attacks.pgd import PGD_Linf, GA_PGD
from attacks.earlystop import Early_PGD


parser = argparse.ArgumentParser(description='PyTorch Adversarially Robust Training in SemiSuperversed Learning')

########################## model setting ##########################
parser.add_argument('--depth', type=int, default=28, help='wideresnet depth factor')
parser.add_argument('--widen_factor', type=int, default=5, help='wideresnet widen factor')
parser.add_argument('--activation', type=str, default= 'relu', choices=['relu', 'leaky', 'silu'], help='choice of activation')
parser.add_argument('--model', type=str, default= 'wideresnet', help='architecture of model') #, choices=['resnet18, wideresnet'] : invalid choice

########################## optimization setting ##########################
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restayts)')
parser.add_argument('--lb_batch_size', default=64, type=int, metavar='N', help='train batchsize')
parser.add_argument('--ul_batch_size', default=128, type=int, metavar='N', help='train batchsize')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--wd', default=5e-4, type=float, metavar='WD', help='weight decay')
parser.add_argument('--lr_scheduler', type=str, default= 'MultiStep', choices=['MultiStep', 'Cosine', 'Cyclic'], help='learning rate scheduling')
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
parser.add_argument('--dataset', type=str, default= 'cifar10', choices=['cifar10', 'cifar100', 'stl10', 'svhn'], help='benchmark dataset')
parser.add_argument('--num_labels', type=int, default= 4000, help='number of labels in semi-supervised learning')
parser.add_argument('--num_classes', type=int, default= 10, help='number of class of dataset')
parser.add_argument('--save_dir', type=str, default='/home/ydy0415/data/experiments/semi-arow', help='save directory')

########################## attack setting ##########################
parser.add_argument('--train_attack', metavar='METHOD', default='pgd_linf', choices=['pgd_linf', 'gapgd_linf'], help=' attack method')
parser.add_argument('--perturb_loss', metavar='LOSS', default='kl', choices=['ce','kl','revkl','js'], help='perturbation loss for adversarial examples')
parser.add_argument('--eps', type=float, default=8, help= 'maximum of perturbation magnitude' )
parser.add_argument('--train_numsteps', type=int, default=10, help= 'train PGD number of steps')
parser.add_argument('--test_numsteps', type=int, default=10, help= 'test PGD number of steps')
parser.add_argument('--random_start', action='store_false', default=True, help='PGD use random start (default: on)')
parser.add_argument('--bn_mode', metavar='BN', default='eval', choices=['eval', 'train'], help='batch normalization mode of attack')

########################## loss setting ##########################
parser.add_argument('--algo', metavar='ALGO', default='semi-arow', choices=['semi-trades', 'rst', 'uat-pp', 'uat-ft', 'fixmatch', 'awr', 'srst-awr', 'kd'], help='surrogate loss function to optimize')
parser.add_argument('--teacher', metavar='TEACHER', default='none', choices=['fixmatch', 'sup', 'none'], help='surrogate loss function to optimize')
parser.add_argument('--vat', action='store_true', help='vat usage flag (default: off)')
parser.add_argument('--tau', type=float, default=1.0, help='temperature of knowledge distillation')
parser.add_argument('--smooth', type=float, default=0.2, help='alpha of label smoothing')
parser.add_argument('--lamb', type=float, default=3., help='coefficient of rob_loss')
parser.add_argument('--gamma', type=float, default=1., help='coefficient of knowledge_loss')
parser.add_argument('--beta', type=float, default=0.5, help='coefficient of convex weighting')
parser.add_argument('--eta', type=float, default=0.95, help='threshold of selecting samples')

########################## SWA setting ##########################
parser.add_argument('--swa', action='store_true', default=False, help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=51, metavar='N', help='SWA start epoch number (default: 50)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N', help='SWA model collection frequency/cycle length in epochs (default: 1)')

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

best_acc = 0  # best val accuracy
attack_best_acc = 0
tolerance = 0 # tolerance

test_acc = 0
attack_test_acc =0

def main(args):
    global best_acc
    global attack_best_acc
    global tolerance
    
    global test_acc
    global attack_test_acc
    
    args.save_dir += f'/{args.dataset}/'
    args.save_dir += str(date.today().strftime('%Y%m%d')[2:])
    
    out_directory = f'{args.save_dir}/{args.train_attack}'
    
    if args.model == "wideresnet":
        out_directory += f'_wrn-{args.depth}-{args.widen_factor}'
        
    elif args.model == "resnet18":
        out_directory += '_resnet18'
    
    out_directory += f'_algo-{args.algo}_perturbloss-{args.perturb_loss}_eps-{args.eps}_lrsche-{args.lr_scheduler}'
    
    out_directory += f'_numlabels-{args.num_labels}'
    
    if args.teacher:
        out_directory += f'_teacher-{args.teacher}'
    # first penalty   
    #['new-arow', 'trades', 'new-cow', 'new-arow-cow', 'fat-trades', 'vat']
    if args.algo in ['semi-arow', 'semi-trades', 'semi-cow', 'semi-arc', 'rst', 'uat-pp', 'conv', 'arow', 'arc', 'kd-arc', 'awr', 'conv-smooth-kd', 'arow-kd', 'arow-soft',  'arow-smooth', 'arow-smooth-kd']:
        out_directory += f'_lamb-{args.lamb}'
        
    if args.algo in ['semi-arow', 'semi-trades', 'semi-cow', 'semi-arc', 'kd-arc', 'conv-smooth-kd', 'arow-kd','arow-smooth-kd']:
        out_directory += f'_gamma-{args.gamma}'
        
    if args.algo in ['semi-arow', 'semi-cow', 'semi-arc', 'conv', 'arow', 'kd-arc', 'awr', 'conv-smooth-kd', 'arow-kd', 'arow-smooth-kd']:
        out_directory += f'_smooth-{args.smooth}'
        
    if args.algo in ['awr', 'conv-smooth-kd']:
        out_directory += f'_beta-{args.beta}'    
        
    if args.algo in ['semi-trades', 'semi-arow', 'semi-cow', 'semi-arc', 'kd-arc', 'conv-smooth-kd', 'arow-kd']:
        out_directory += f'_tau-{args.tau}'
        
    if args.vat:
        out_directory += f'_vat-{args.vat}'
        
    if args.swa:
        out_directory += f'_swa-{args.swa}'
    
    out_directory += f'_seed-{args.seed}'
        
    if args.add_name != '':
        out_directory += f'_{args.add_name}'
    
    if not os.path.isdir(out_directory):
        mkdir_p(out_directory)
    
    # Data
    print('==> Preparing ' + str(args.dataset))
    
    train_dataset = SSL_Dataset(args, name=args.dataset, train=True, num_classes=args.num_classes, data_dir=args.data_dir)
    
    lb_dataset, ulb_dataset = train_dataset.get_ssl_dset(num_labels=args.num_labels, index=None, include_lb_to_ulb=True, strong_transform=True, onehot=False)
    
    test_transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)
    elif args.dataset == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=test_transform)
    elif args.dataset == 'svhn':
        test_dataset = torchvision.datasets.SVHN(root=args.data_dir, split= 'test', download=True, transform=test_transform) 
    elif args.dataset == 'stl10':
        test_dataset = torchvision.datasets.STL10(root=args.data_dir, split= 'test', download=True, transform=test_transform) 
    else:
        raise ValueError("dataset error.")
    
    
    labeled_trainloader = torch.utils.data.DataLoader(lb_dataset, batch_size=args.lb_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    unlabeled_trainloader = torch.utils.data.DataLoader(ulb_dataset, batch_size=args.ul_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.ul_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # Model
    
    def create_model(swa=False):
        if args.model == 'wideresnet':
            print("==> creating WideResNet" + str(args.depth) + '-' + str(args.widen_factor))
            if swa:
                swa_model = wrn_models.WideResNet(first_stride = 2 if args.dataset in ['stl10'] else 1,
                                                  num_classes  = args.num_classes,
                                                  depth        = args.depth,
                                                  widen_factor = args.widen_factor,
                                                  activation   = args.activation).cuda(args.gpu)
                model = wrn_models.WideResNet(first_stride = 2 if args.dataset in ['stl10'] else 1,
                                                  num_classes  = args.num_classes,
                                                  depth        = args.depth,
                                                  widen_factor = args.widen_factor,
                                                  activation   = args.activation).cuda(args.gpu)

                return swa_model, model

            else:
                model = wrn_models.WideResNet(first_stride = 2 if args.dataset in ['stl10'] else 1,
                                                  num_classes  = args.num_classes,
                                                  depth        = args.depth,
                                                  widen_factor = args.widen_factor,
                                                  activation   = args.activation).cuda(args.gpu)
                
                return model
        
        elif args.model == 'resnet18':
            print("==> creating ResNet18")
            if swa:
                swa_model = res_models.resnet('resnet18', input_channel, num_classes=10).cuda(args.gpu)
                model = res_models.resnet('resnet18', input_channel, num_classes=10).cuda(args.gpu)

                return swa_model, model
            
            else:
                model = res_models.resnet('resnet18', input_channel, num_classes=10).cuda(args.gpu)
            
                return model
    
    if args.swa:
        swa_model, model = create_model(args.swa)
        swa_n = 0
    else:
        model = create_model(args.swa)
    
    teacher_model = wrn_models.WideResNet(first_stride = 2 if args.dataset in ['stl10'] else 1,
                                          num_classes  = args.num_classes,
                                          depth        = args.depth,
                                          widen_factor = args.widen_factor,
                                          activation   = args.activation).cuda(args.gpu)
    
    
    checkpoint = torch.load(f'/home/ydy0415/data/teacher/{args.dataset}/wrn-{args.depth}-{args.widen_factor}_teacher-{args.teacher}_numlabels-{args.num_labels}.pth.tar'
                            , map_location='cuda:' + str(args.gpu))
    #teacher_model.load_state_dict(checkpoint['ema_state_dict'])
    teacher_model.load_state_dict(checkpoint['state_dict'])
    teacher_model.eval()
    del checkpoint
    torch.cuda.empty_cache()
    print("==> The teacher-model is loaded.")
        
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    criterion = nn.CrossEntropyLoss()
    
    if args.train_attack == 'pgd_linf':
        train_attack = PGD_Linf(model = model,
                                epsilon = args.eps/255,
                                step_size = (args.eps/4)/255,
                                num_steps=args.train_numsteps,
                                random_start=args.random_start,
                                criterion=args.perturb_loss,
                                bn_mode = args.bn_mode,
                                train = True,
                                vat=args.vat)
        
    elif args.train_attack == 'gapgd_linf':
        train_attack = GA_PGD(model = model,
                              epsilon=args.eps/255,
                              step_size=(args.eps/4)/255,
                              num_steps=args.train_numsteps,
                              random_start=args.random_start,
                              criterion=args.perturb_loss,
                              bn_mode = args.bn_mode,
                              train = True)
        
    test_attack = PGD_Linf(model = model,
                           epsilon=args.eps/255,
                           step_size=(args.eps/4)/255,
                           num_steps=args.test_numsteps,
                           random_start=args.random_start,
                           criterion='ce',
                           bn_mode = args.bn_mode,
                           train = False)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    
    if args.lr_scheduler == "MultiStep":
        if args.swa:
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150], gamma=0.1) #cifar10
        else:
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)
            
    elif args.lr_scheduler == "Cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == "Cyclic":
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.swa_start)
        scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.03, step_size_up=1, step_size_down = 3,gamma=1)

    # Resume
    title = args.dataset
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.save_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        test_acc = checkpoint['test_acc']
        attack_best_acc = checkpoint['attack_test_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(out_directory, 'log.txt'), title=title, resume=True)
        if args.swa:
            swa_model.load_state_dict(checkpoint['swa_state_dict'])
            swa_n = checkpoint['swa_n']
        del checkpoint
        torch.cuda.empty_cache()
        
    else:
        logger = Logger(os.path.join(out_directory, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'Train Loss', 'Test Loss', 'Test Acc.', 'Attack Loss' , 'Attack Acc.'])
    
    # Train and val
    for epoch in range(args.start_epoch, args.epochs + 1):

        print('\n'+args.train_attack +' Epoch: [%d | %d] LR: %.5f Tol: %d Best ts acc: %.2f Best_att_acc: %.2f ' % (epoch, args.epochs, optimizer.param_groups[0]['lr'], tolerance, best_acc, attack_best_acc))
        
        if args.resume:
            validate(test_loader, swa_model, criterion, use_cuda, mode='Attack_test', attack=test_attack)
        
        train_loss = train(args,
                           labeled_trainloader,
                           unlabeled_trainloader,
                           epoch,
                           model,
                           optimizer,
                           use_cuda,
                           attack=train_attack,
                           teacher_model=teacher_model)
        
        if args.swa and epoch == args.swa_start:
            test_attack =  PGD_Linf(model=swa_model,
                                    epsilon=args.eps/255,
                                    step_size=(args.eps/4)/255,
                                    num_steps=args.test_numsteps,
                                    random_start=args.random_start,
                                    criterion='ce',
                                    bn_mode = args.bn_mode,
                                    train = False)
        
        if args.swa and epoch >= args.swa_start and (epoch - args.swa_start) % args.swa_c_epochs == 0:
            moving_average(swa_model, model, 1.0 / (swa_n + 1))
            swa_n += 1
            if epoch % 2 == 0:
                bn_update(unlabeled_trainloader, swa_model)
                test_loss, test_acc = validate(test_loader,
                                               swa_model,
                                               criterion,
                                               use_cuda,
                                               mode='Test')
                
                attack_test_loss, attack_test_acc = validate(test_loader,
                                                             swa_model,
                                                             criterion,
                                                             use_cuda,
                                                             mode='Attack_test',
                                                             attack=test_attack)
                
                logger.append([round(epoch), train_loss, test_loss, test_acc, attack_test_loss,  attack_test_acc])
            
        else:
            if epoch == 1 or epoch % 5 == 0 or epoch >= args.epochs - 20 :
                test_loss, test_acc = validate(test_loader,
                                               model,
                                               criterion,
                                               use_cuda,
                                               mode='Test')
                
                attack_test_loss, attack_test_acc = validate(test_loader,
                                                             model,
                                                             criterion,
                                                             use_cuda,
                                                             mode='Attack_test',
                                                             attack=test_attack)
                
                logger.append([round(epoch), train_loss, test_loss, test_acc, attack_test_loss,  attack_test_acc])
            
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
        is_attack_best = attack_test_acc > attack_best_acc
        attack_best_acc = max(attack_test_acc, attack_best_acc)
        
        #is_best = test_acc >= best_acc
        #best_acc = max(test_acc, best_acc)
        
        #if is_best:
        #    best_acc = test_acc
        if is_attack_best:
            attack_best_acc, best_acc = attack_test_acc, test_acc
        
        if args.swa:
            if epoch >= args.swa_start and (epoch - args.swa_start) % args.swa_c_epochs == 0 and is_attack_best:
                save_checkpoint(out_directory, epoch,
                filename='robust_best.pth.tar',
                swa_state_dict = swa_model.state_dict(),
                swa_n = swa_n,
                state_dict = model.state_dict(),
                test_acc =  test_acc,
                attack_test_acc = attack_test_acc,
                optimizer = optimizer.state_dict()
                )
            elif epoch < args.swa_start and is_attack_best:
                save_checkpoint(out_directory, epoch, 
                filename='robust_best.pth.tar',
                state_dict = model.state_dict(),
                test_acc = test_acc,
                attack_test_acc = attack_test_acc,
                optimizer = optimizer.state_dict()
                )
            elif epoch == args.epochs:
                save_checkpoint(out_directory, epoch, 
                filename='last.pth.tar',
                swa_state_dict = swa_model.state_dict(),
                test_acc = test_acc,
                attack_test_acc = attack_test_acc
                )
                
        elif not args.swa:
            if is_attack_best:
                save_checkpoint(out_directory, epoch, 
                    filename='robust_best.pth.tar',       
                    state_dict = model.state_dict(),
                    test_acc = test_acc,
                    attack_test_acc = attack_test_acc,
                    optimizer = optimizer.state_dict()
                    )
            '''
            if epoch >= args.epochs-5:
                save_checkpoint(out_directory, epoch, 
                    filename= str(epoch) + '_model.pth.tar',
                    state_dict = model.state_dict(),
                    test_acc = test_acc,
                    attack_test_acc = attack_test_acc,
                    optimizer = optimizer.state_dict()
                    )
            '''    
            if epoch == args.epochs:
                save_checkpoint(out_directory, epoch, 
                filename='last.pth.tar',
                state_dict = model.state_dict(),
                test_acc = test_acc,
                attack_test_acc = attack_test_acc
                )


        if is_attack_best:
            tolerance = 0
        else:
            tolerance += 1

    logger.close()

    print('Best test acc:')
    print(best_acc)

    print('Best attack acc:')
    print(attack_best_acc)

def train(args, labeled_trainloader, unlabeled_trainloader, epoch, model, optimizer, use_cuda, attack, teacher_model=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    sup_losses = AverageMeter()
    reg_losses = AverageMeter()
    know_losses = AverageMeter()
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
            inputs_lb, targets_lb, inputs_ul_w = inputs_lb.cuda(), targets_lb.cuda(non_blocking=True), inputs_ul_w.cuda()
                
        with amp_cm():
           
               
            if args.algo == "semi-trades":
                sup_loss, reg_loss, know_loss = Semi_TRADES_loss(inputs_lb,
                                                                 targets_lb,
                                                                 inputs_ul_w,
                                                                 model,
                                                                 attack,
                                                                 args.tau,
                                                                 teacher_model)
                
                reg_loss = args.lamb * reg_loss
                know_loss = args.gamma * know_loss 
                loss = sup_loss + reg_loss + know_loss
                
            elif args.algo == "rst":
                sup_loss, reg_loss = RST_loss(inputs_lb,
                                              targets_lb,
                                              inputs_ul_w,
                                              model,
                                              attack,
                                              teacher_model)
                #                def RST_loss(inputs_lb, targets_lb, inputs_ul_w, model, attack, teacher_model):
                reg_loss = args.lamb * reg_loss
                know_loss = torch.tensor(0.)
                loss = sup_loss + reg_loss
            
            elif args.algo == "uat-ft":
                if args.perturb_loss not in ["ce"]:
                     raise ValueError("perturb loss must be ce.")
                if args.teacher not in ["sup"]:
                     raise ValueError("teacher should be supervised model.")
                        
                sup_loss = UAT_FT_loss(inputs_lb,
                                       targets_lb,
                                       inputs_ul_w,
                                       model,
                                       attack,
                                       teacher_model)
                #                def UAT_PP_loss(inputs_lb, targets_lb, inputs_ul, model, attack, teacher_model):
                reg_loss =  torch.tensor(0.)
                know_loss = torch.tensor(0.)
                loss = sup_loss
            
            
            elif args.algo == "uat-pp":
                if args.perturb_loss not in ["ce"]:
                     raise ValueError("perturb loss must be ce.")
                        
                sup_loss, reg_loss = UAT_PP_loss(inputs_lb,
                                                 targets_lb,
                                                 inputs_ul_w,
                                                 model,
                                                 attack,
                                                 teacher_model)
                
                reg_loss = args.lamb * reg_loss
                know_loss = torch.tensor(0.)
                loss = sup_loss + reg_loss

                
            elif args.algo == "awr":
                sup_loss, reg_loss = AWR_loss(inputs_lb,
                                              targets_lb,
                                              inputs_ul_w,
                                              model,
                                              attack,
                                              args.smooth,
                                              args.beta,
                                              teacher_model)
                
                reg_loss = args.lamb * reg_loss
                know_loss = torch.tensor(0.)
                loss = sup_loss + reg_loss
            
           
            elif args.algo == "srst-awr":
                sup_loss, reg_loss, know_loss = SRST_AWR_KD_loss(inputs_lb,
                                                                    targets_lb,
                                                                    inputs_ul_w,
                                                                    model,
                                                                    attack,
                                                                    args.smooth,
                                                                    args.beta,
                                                                    args.tau,
                                                                    teacher_model)
                
                reg_loss = args.lamb * reg_loss
                know_loss = args.gamma * know_loss
                loss = sup_loss + reg_loss + know_loss
                
        # record loss
        
        sup_losses.update(sup_loss.item(), inputs_lb.size(0))
        reg_losses.update(reg_loss, inputs_ul_w.size(0))
        know_losses.update(know_loss, inputs_ul_w.size(0))
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
        bar.suffix  = '({batch:>3}/{size:>3}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Sup_loss: {sup_loss:.4f} | Reg_loss: {reg_loss:.4f} | Know_loss: {know_loss:.4f}  | Tot loss:{loss:.4f}'.format(
                    batch   = batch_idx + 1,
                    size    = len(unlabeled_trainloader),
                    data    = data_time.avg,
                    bt      = batch_time.avg,
                    total   = bar.elapsed_td,
                    eta     = bar.eta_td,
                    sup_loss=sup_losses.avg,
                    reg_loss=reg_losses.avg,
                    know_loss=know_losses.avg,
                    loss=losses.avg
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
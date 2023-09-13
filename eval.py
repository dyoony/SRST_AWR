from __future__ import print_function

import time
import os
import csv
import shutil
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn. functional as F
import torch.backends.cudnn as cudnn
import argparse


from PIL import Image
import torchvision
from torchvision import models, transforms
from torchvision.utils import save_image
from torchvision import datasets, transforms
import torch.utils.data as data

from attacks.pgd import PGD_Linf, PGD_L2
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p
#from tqdm.notebook import tqdm
from autoattack import AutoAttack

import models.wideresnet as wrn_models
import models.resnet as res_models

import load_data.datasets as dataset

parser = argparse.ArgumentParser(description='Test the robustness to adversarial attack')

# ########################## basic settin
parser.add_argument('--seed', type=int, default=0, help='manual seed')
parser.add_argument('--gpu', default=0, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='test batchsize')


######################### Dataset #############################
parser.add_argument('--dataset', type=str, default= 'cifar10', choices=['cifar10', 'cifar100', 'stl10', 'fmnist', 'svhn'], help='benchmark dataset')
parser.add_argument('--data_dir', default='/home/ydy0415/data/datasets', help='Directory of dataset')
parser.add_argument('--num_labels', type=int, default= 4000, help='number of labels in semi-supervised learning')
parser.add_argument('--num_classes', type=int, default= 10, help='number of class of dataset')

######################### Robust Evaluation Setting #############################
parser.add_argument('--attack_method', metavar='METHOD', default='both', choices=['autoattack', 'pgd_linf' , 'both','pgd_l2', 'fgsm'], help=' attack method')
parser.add_argument('--eps', type=float, default=8, help= 'maximum of perturbation magnitude' )
parser.add_argument('--test_numsteps', type=int, default=20, help= 'test PGD number of steps')
parser.add_argument('--random_start', action='store_false', help='PGD use random start')
parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--out', default='/home/ydy0415/data/experiments/semi-arow/cifar10/results', help='Directory to output the result')
parser.add_argument('--bn_mode', metavar='BN', default='eval', choices=['eval', 'train'], help='batch normalization mode of attack')

########################## Model Setting ##########################
parser.add_argument('--model', type=str, default= 'wideresnet', help='architecture of model') #, choices=['resnet18, wideresnet'] : invalid choice
parser.add_argument('--depth', type=int, default=28, help='wideresnet depth factor')
parser.add_argument('--widen_factor', type=int, default=5, help='wideresnet widen factor')
parser.add_argument('--activation', type=str, default= 'relu', choices=['relu', 'leaky', 'silu'], help='choice of activation')
parser.add_argument('--model_dir', default='', help='Directory of model saved')

########################## Misc ##########################
parser.add_argument('--add_name', default='', type=str, help='add_name')



args = parser.parse_args()
print (args)

state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.cuda.set_device(args.gpu)
use_cuda = torch.cuda.is_available()

if args.dataset in ['cifar10', 'cifar100', 'stl10', 'svhn']:
    input_channel = 3
elif args.dataset in ['fmnist']:
    input_channel = 1


def main():
    
    out_directory= args.out
    data_directory = args.data_dir
    
    # Data
    print('==> Preparing ' + str(args.dataset))
    
    
    
    data_directory = args.data_dir
    
    test_transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)
    elif args.dataset == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=test_transform)
    elif args.dataset == 'svhn':
        test_dataset = torchvision.datasets.SVHN(root=args.data_dir, split= 'test', download=True, transform=test_transform) 
    elif args.dataset == 'stl10':
        test_dataset = torchvision.datasets.STL10(root=args.data_dir, split= 'test', download=True, transform=test_transform) 
    # Model
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    def load_model(args):
        if args.model == 'wideresnet':
            print("==> creating WideResNet" + str(args.depth) + '-' + str(args.widen_factor))
            model = wrn_models.WideResNet(first_stride = 2 if args.dataset in ['stl10'] else 1,
                                          num_classes=100 if args.dataset=='cifar100' else 10,
                                          depth=args.depth,
                                          widen_factor=args.widen_factor,
                                          activation=args.activation).cuda(args.gpu)
            checkpoint = torch.load(args.model_dir + '/robust_best.pth.tar', map_location= 'cuda:' + str(args.gpu))
            if args.swa:
                model.load_state_dict(checkpoint['swa_state_dict'])   
            else:
                model.load_state_dict(checkpoint['state_dict'])
        
        elif args.model == 'resnet18':
            print("==> creating ResNet18")
            model = res_models.resnet('resnet18', input_channel, num_classes=100 if args.dataset=='cifar100' else 10).cuda(args.gpu)
            checkpoint = torch.load(args.model_dir + '/robust_best.pth.tar', map_location= 'cuda:' + str(args.gpu))
            #checkpoint = torch.load(args.model_dir + '/last.pth.tar', map_location= 'cuda:' + str(args.gpu))
            if args.swa:
                model.load_state_dict(checkpoint['swa_state_dict'])              
            elif args.ema:
                model.load_state_dict(checkpoint['ema_state_dict'])    
            else:
                model.load_state_dict(checkpoint['state_dict'])
            del checkpoint
            torch.cuda.empty_cache()
            
        elif args.model == 'pre-resnet18':
            print("==> creating Pre-ResNet18")
            model = pre_res_models.preact_resnet('preact-resnet18', input_channel, num_classes=10).cuda(args.gpu)
            checkpoint = torch.load(args.model_dir + '/robust_best.pth.tar', map_location= 'cuda:' + str(args.gpu))
            #checkpoint = torch.load(args.model_dir + '/last.pth.tar', map_location= 'cuda:' + str(args.gpu))
            if args.swa:
                model.load_state_dict(checkpoint['swa_state_dict'])              
            elif args.ema:
                model.load_state_dict(checkpoint['ema_state_dict'])    
            else:
                model.load_state_dict(checkpoint['state_dict'])
            del checkpoint
            torch.cuda.empty_cache()
                
        return model
    
    model = load_model(args)
    
    criterion = nn.CrossEntropyLoss()
    #kl_div= nn.KLDivLoss()
    #cam_criterion = nn.MSELoss()
    
    if args.attack_method == 'pgd_l2':
        test_attack = PGD_L2(model=model, epsilon=args.eps/255, step_size=(args.eps/10)/255, num_steps=args.test_numsteps, random_start=args.random_start, train=False)
    elif args.attack_method == 'pgd_linf':
        test_attack= PGD_Linf(model=model, epsilon=args.eps/255, step_size=(args.eps/4)/255, num_steps=args.test_numsteps, random_start=args.random_start, criterion='ce',
                              bn_mode = args.bn_mode, train = False)
    elif args.attack_method == 'fgsm':
        test_attack = FGSM(model=model, epsilon=args.eps/255)
    elif args.attack_method == 'autoattack':
        auto_attack = AutoAttack(model, norm='Linf', eps=args.eps/255, version='standard', verbose=False)
        #auto_attack.attacks_to_run = ['apgd-ce', 'apgd-t']
        auto_attack.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab', 'square']
        #auto_attack.attacks_to_run = ['fab', 'square']
    elif args.attack_method == 'both':
        test_attack= PGD_Linf(model=model, epsilon=args.eps/255, step_size=(args.eps/4)/255, num_steps=args.test_numsteps, random_start=args.random_start, criterion='ce',
                              bn_mode = args.bn_mode, train = False)
        auto_attack = AutoAttack(model, norm='Linf', eps=args.eps/255, version='standard', verbose=False)
        auto_attack.attacks_to_run = ['apgd-ce', 'apgd-t']
        #auto_attack.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab']
        #auto_attack.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab', 'square']
    
    resname = args.out + 'log.csv'
    if not os.path.exists(resname):
        os.makedirs(args.out, exist_ok=True)
        with open(resname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if args.attack_method == 'both':
                logwriter.writerow(['Model', 'Test Loss', 'Test Acc.' , 'Attack Acc.(PGD_linf)', 'Attack Acc.(AutoAttack)'])
            else:
                logwriter.writerow(['Model', 'Test Loss', 'Test Acc.' , 'Attack Acc.(' + str(args.attack_method)+ ')'])
            

    cudnn.benchmark = True
    
    print("==> Starting test for " + str(args.attack_method))
    test_loss, test_acc = validate(test_loader, model, criterion, use_cuda, mode='Test')
    
    if args.attack_method == 'both':
        _, pgd_test_acc = validate(test_loader, model, criterion, use_cuda, mode='PGD_attack', pgd_attack=test_attack)
        _, aa_test_acc  = validate(test_loader, model, criterion, use_cuda, mode='Autoattack', pgd_attack=None, autoattack=auto_attack)
        
    elif args.attack_method=='autoattack':
        _, aa_test_acc  = validate(test_loader, model, criterion, use_cuda, mode='Autoattack', pgd_attack=None, autoattack=auto_attack)
        
    else:
        _, pgd_test_acc = validate(test_loader, model, criterion, use_cuda, mode='PGD_attack', pgd_attack=test_attack)
                            #validate(test_loader, swa_model, criterion, epoch, use_cuda, mode='Attack_test', attack=test_attack)
            #validate(valloader, model, criterion, use_cuda, attack, autoattack=None, adversary=None)
    
    
    #################### Write results ####################
    model_name=args.model_dir.split('/')[-1]
    
    with open(resname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        if args.attack_method == 'both':
            logwriter.writerow([model_name, test_loss, test_acc, pgd_test_acc, aa_test_acc])
        elif args.attack_method == 'autoattack':
            logwriter.writerow([model_name, test_loss, test_acc, aa_test_acc]) 
        else:
            logwriter.writerow([model_name, test_loss, test_acc, pgd_test_acc])

    print('Test acc:{}'.format(test_acc))
    if args.attack_method == 'both':
        print('PGD-attack Acc:{}'.format(pgd_test_acc))
        print('Autoattack Acc:{}'.format(aa_test_acc))
    elif args.attack_method == 'autoattack':
        print('Autoattack Acc:{}'.format(aa_test_acc))
    else:
        print('PGD-attack Acc:{}'.format(pgd_test_acc))

def validate(valloader, model, criterion, use_cuda, mode, pgd_attack=None, autoattack=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    bar = Bar('{mode:>12}'.format(mode=mode), max=len(valloader))
   
    for batch_idx, (inputs, targets) in enumerate(valloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

        if not autoattack and pgd_attack:
            adv_inputs, _ = pgd_attack.perturb(inputs, targets)
            outputs = model(adv_inputs)
        elif not autoattack and not pgd_attack:
            outputs = model(inputs)
        elif autoattack and not pgd_attack:
            adv_inputs = autoattack.run_standard_evaluation(inputs, targets, bs=args.batch_size)
            #print((adv_inputs - inputs).shape, (adv_inputs-inputs).max())
            outputs = model(adv_inputs)
        else:
            raise ValueError("You should select one method.")

        loss = criterion(outputs, targets)

    # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if not autoattack and not pgd_attack:
            del inputs, outputs, targets
        else:
            del inputs, outputs, targets, adv_inputs
        torch.cuda.empty_cache()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch:>3}/{size:>3}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
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


if __name__ == '__main__':
    main()
"""
Datasets with unlabeled (or pseudo-labeled) data
"""

from torchvision.datasets import CIFAR10, SVHN
from torch.utils.data import Sampler, Dataset
from torchvision import transforms
from .augmentation.randaugment import RandAugment

from PIL import Image
import torch
import numpy as np


import os
import pickle
import copy
import logging

from .cifar10 import load_cifar10
from .fmnist import load_fmnist
from .svhn import load_svhn
from .data_utils import get_onehot

DATASETS = ['cifar10', 'fmnist', 'svhn']

def load_data(data_dir, dataset, batch_size=128, batch_size_test=128, num_workers=4, use_augmentation=True, shuffle_train=True, validation=False):
    """
    Returns train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        batch_size (int): batch size for training.
        batch_size_test (int): batch size for validation.
        num_workers (int): number of workers for loading the data.
        use_augmentation (bool): whether to use augmentations for training set.
        shuffle_train (bool): whether to shuffle training set.
        aux_data_filename (str): path to unlabelled data.
        unsup_fraction (float): fraction of unlabelled data per batch.
        validation (bool): if True, also returns a validation dataloader for unspervised cifar10 (as in Gowal et al, 2020).
    """
    if dataset == 'cifar10':
        train_dataset, test_dataset = load_cifar10(data_dir=data_dir, use_augmentation=use_augmentation)
    elif dataset == 'fmnist':
        train_dataset, test_dataset = load_fmnist(data_dir=data_dir, use_augmentation=use_augmentation)
    elif dataset == 'svhn':
        train_dataset, test_dataset = load_svhn(data_dir=data_dir, use_augmentation=False)
    else:
        raise ValueError("Invalid dataset") 
    
    if validation:
        num_train_samples = len(train_dataset)
        val_dataset = torch.utils.data.Subset(train_dataset, torch.arange(0, 1024))
        train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(1024, num_train_samples))
    
    
    pin_memory = torch.cuda.is_available()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, 
                                                   num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, 
                                                  num_workers=num_workers, pin_memory=pin_memory)
    if validation:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False, 
                                                         num_workers=num_workers, pin_memory=pin_memory)
        
    if validation:
        return train_dataset, test_dataset, val_dataset, train_dataloader, test_dataloader, val_dataloader
    
    return train_dataset, test_dataset, train_dataloader, test_dataloader


class SemiSupervisedDataset(Dataset):
    def __init__(self,
                 base_dataset='cifar10',
                 take_amount=None,
                 take_amount_seed=13,
                 add_svhn_extra=False,
                 aux_data_filename=None,
                 add_aux_labels=False,
                 aux_take_amount=None,
                 train=False,
                 **kwargs):
        """A dataset with auxiliary pseudo-labeled data"""
        # take_amount : labeled data set size
        # add_svhn_extra : svhn extra data
        # add_aux_labels : whther to add pseduo label with unlabeled data
        # aux_take_amount : unlabeled data set size

        if base_dataset == 'cifar10':
            self.dataset = CIFAR10(train=train, **kwargs)
        elif base_dataset == 'svhn':
            if train:
                self.dataset = SVHN(split='train', **kwargs)
            else:
                self.dataset = SVHN(split='test', **kwargs)
            # because torchvision is annoying
            self.dataset.targets = self.dataset.labels
            self.targets = list(self.targets)

            if train and add_svhn_extra:
                svhn_extra = SVHN(split='extra', **kwargs)
                self.data = np.concatenate([self.data, svhn_extra.data])
                self.targets.extend(svhn_extra.labels)
        else:
            raise ValueError('Dataset %s not supported' % base_dataset)
        self.base_dataset = base_dataset
        self.train = train

        if self.train:
            if take_amount is not None:
                rng_state = np.random.get_state()
                np.random.seed(take_amount_seed)
                take_inds = np.random.choice(len(self.sup_indices),
                                             take_amount, replace=False)
                np.random.set_state(rng_state)

                logger = logging.getLogger()
                logger.info('Randomly taking only %d/%d examples from training'
                            ' set, seed=%d, indices=%s',
                            take_amount, len(self.sup_indices),
                            take_amount_seed, take_inds)
                self.targets = self.targets[take_inds]
                self.data = self.data[take_inds]

            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            if aux_data_filename is not None:
                aux_path = os.path.join(kwargs['root'], aux_data_filename)
                print("data from %s" % aux_path)
                if aux_data_filename == 'ti_500K_pseudo_labeled.pickle':
                    with open(aux_path, 'rb') as f:
                        aux = pickle.load(f)
                    aux_data = aux['data']
                    aux_targets = aux['extrapolated_targets']
                elif aux_data_filename == 'cifar10_ddpm.npz':
                    #with open(aux_path, 'rb') as f:
                    aux = np.load(aux_path)
                    aux_data = aux['image']
                    aux_targets = aux['label']
                    del aux
                elif aux_data_filename == 'cifar10_ddpm_fixmatch_4000.npz':
                    #with open(aux_path, 'rb') as f:
                    #    aux = np.load(f)
                    aux = np.load(aux_path)
                    aux_data = aux['image']
                    aux_targets = aux['label']
                    del aux
                elif aux_data_filename == 'cifar10_ddpm_sup_4000.npz':
                    #with open(aux_path, 'rb') as f:
                    aux = np.load(aux_path)
                    aux_data = aux['image']
                    aux_targets = aux['label']
                    del aux
                else:
                    raise ValueError('check dataset')
                    
                orig_len = len(self.data)
                
                if aux_take_amount is not None:
                    rng_state = np.random.get_state()
                    np.random.seed(take_amount_seed)
                    take_inds = np.random.choice(len(aux_data),
                                                 aux_take_amount, replace=False)
                    np.random.set_state(rng_state)

                    logger = logging.getLogger()
                    logger.info(
                        'Randomly taking only %d/%d examples from aux data'
                        ' set, seed=%d, indices=%s',
                        aux_take_amount, len(aux_data),
                        take_amount_seed, take_inds)
                    aux_data = aux_data[take_inds]
                    aux_targets = aux_targets[take_inds]

                self.data = np.concatenate((self.data, aux_data), axis=0)

                if not add_aux_labels:
                    self.targets.extend([-1] * len(aux_data))
                else:
                    self.targets.extend(aux_targets)
                # note that we use unsup indices to track the labeled datapoints
                # whose labels are "fake"
                self.unsup_indices.extend(
                    range(orig_len, orig_len+len(aux_data)))

            logger = logging.getLogger()
            logger.info("Training set")
            logger.info("Number of training samples: %d", len(self.targets))
            logger.info("Number of supervised samples: %d",
                        len(self.sup_indices))
            logger.info("Number of unsup samples: %d", len(self.unsup_indices))
            logger.info("Label (and pseudo-label) histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of training data: %s", np.shape(self.data))

        # Test set
        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            logger = logging.getLogger()
            logger.info("Test set")
            logger.info("Number of samples: %d", len(self.targets))
            logger.info("Label histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of data: %s", np.shape(self.data))

    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, value):
        self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets  # because torchvision is annoying
        return self.dataset[item]

    def __repr__(self):
        fmt_str = 'Semisupervised Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.dataset.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.dataset.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.dataset.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SemiSupervisedSampler(Sampler):
    """Balanced sampling from the labeled and unlabeled data"""
    def __init__(self, sup_inds, unsup_inds, batch_size, unsup_fraction=0.5,
                 num_batches=None):
        # sup_inds : labeled data index
        # unsup_inds : unlabeled data index
        # unsup_fraction : ratio of labeled data and unlabeled data with pseudo label
        if unsup_fraction is None or unsup_fraction < 0:
            self.sup_inds = sup_inds + unsup_inds
            unsup_fraction = 0.0
        else:
            self.sup_inds = sup_inds
            self.unsup_inds = unsup_inds

        self.batch_size = batch_size
        unsup_batch_size = int(batch_size * unsup_fraction)
        self.sup_batch_size = batch_size - unsup_batch_size

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(
                np.ceil(len(self.sup_inds) / self.sup_batch_size))

        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            sup_inds_shuffled = [self.sup_inds[i]
                                 for i in torch.randperm(len(self.sup_inds))]
            for sup_k in range(0, len(self.sup_inds), self.sup_batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = sup_inds_shuffled[sup_k:(sup_k + self.sup_batch_size)]
                if self.sup_batch_size < self.batch_size:
                    batch.extend([self.unsup_inds[i] for i in
                                  torch.randint(high=len(self.unsup_inds),
                                                size=(self.batch_size - len(batch),),
                                                dtype=torch.int64)])
                # this shuffle operation is very important, without it
                # batch-norm / DataParallel hell ensues
                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches

    

class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot

        self.transform = transform
        if self.is_ulb:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3, 5))
        else:
            self.strong_transform = strong_transform

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """

        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images

        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.is_ulb:
                return idx, img_w, target
            else:
                if self.alg == 'fixmatch':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'flexmatch':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'pimodel':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'pseudolabel':
                    return idx, img_w
                elif self.alg == 'vat':
                    return idx, img_w
                elif self.alg == 'meanteacher':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'uda':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'mixmatch':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return idx, img_w, img_s1, img_s2, img_s1_rot, rotate_v_list.index(rotate_v1)
                elif self.alg == 'fullysupervised':
                    return idx

    def __len__(self):
        return len(self.data)

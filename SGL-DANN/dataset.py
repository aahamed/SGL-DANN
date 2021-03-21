import torch
import torchvision.datasets as dset
import utils
import torch.utils.data as data
from PIL import Image
import os
import numpy as np

# DataLoader for MNIST-M dataset
class MnistmGetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data


def get_data_transforms( dataset_name, args ):
    train_transform, valid_transform = None, None
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    elif dataset_name == 'mnist':
        train_transform, valid_transform = utils._data_transforms_mnist(args)
    elif dataset_name == 'mnistm':
        train_transform, valid_transform = utils._data_transforms_mnistm(args)
    elif dataset_name == 'cifar10_hsv':
        train_transform, valid_transform = utils._data_transforms_cifar10_hsv(args)
    else:
        assert False and f'Unrecognized dataset: {dataset_name}'
    return train_transform, valid_transform

def get_train_dataset( dataset_name, args ):
    # import pdb; pdb.set_trace()
    train_data = None
    train_transform, _ = get_data_transforms( dataset_name, args )
    if dataset_name == 'cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, 
                download=True, transform=train_transform)
    elif dataset_name == 'cifar10':
        train_data = dset.CIFAR10(root=args.data, train=True, 
                download=True, transform=train_transform)
    elif dataset_name == 'mnist':
        train_data = dset.MNIST(
            root=args.data,
            train=True,
            transform=train_transform,
            download=True
        )
    elif dataset_name == 'mnistm':
        mnistm_root = os.path.join( args.data, 'mnist_m' )
        train_list = os.path.join(mnistm_root, 'mnist_m_train_labels.txt')
        train_data = MnistmGetLoader(
            data_root=os.path.join(mnistm_root, 'mnist_m_train'),
            data_list=train_list,
            transform=train_transform
        )
    elif dataset_name == 'cifar10_hsv':
        train_data = dset.CIFAR10(root=args.data, train=True, 
                download=True, transform=train_transform)
    else:
        assert False and f'Unrecognized dataset: {dataset_name}'
    return train_data

def get_dataset( dataset_name, args, train=True ):
    # import pdb; pdb.set_trace()
    t_data = None
    t_transform, _ = get_data_transforms( dataset_name, args )
    if dataset_name == 'cifar100':
        t_data = dset.CIFAR100(root=args.data, train=train, 
                download=True, transform=t_transform)
    elif dataset_name == 'cifar10':
        t_data = dset.CIFAR10(root=args.data, train=train, 
                download=True, transform=t_transform)
    elif dataset_name == 'mnist':
        t_data = dset.MNIST(
            root=args.data,
            train=train,
            transform=t_transform,
            download=True
        )
    elif dataset_name == 'mnistm':
        mnistm_root = os.path.join( args.data, 'mnist_m' )
        if train:
            train_list = os.path.join(mnistm_root, 'mnist_m_train_labels.txt')
            t_data = MnistmGetLoader(
                data_root=os.path.join(mnistm_root, 'mnist_m_train'),
                data_list=train_list,
                transform=t_transform
            )
        else:
            test_list = os.path.join(mnistm_root, 'mnist_m_test_labels.txt')
            t_data = MnistmGetLoader(
                data_root=os.path.join(mnistm_root, 'mnist_m_test'),
                data_list=test_list,
                transform=t_transform )
    elif dataset_name == 'cifar10_hsv':
        t_data = dset.CIFAR10(root=args.data, train=True, 
                download=True, transform=t_transform)
    else:
        assert False and f'Unrecognized dataset: {dataset_name}'
    return t_data
    
def get_train_dataloaders( dataset_name, args ):
    train_data = get_dataset( dataset_name, args, train=True )
    num_train = len( train_data )
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=8 )

    unlabeled_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=8 )

    valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(
          indices[split:num_train]),
      pin_memory=True, num_workers=8 )

    return train_queue, unlabeled_queue, valid_queue

def get_test_dataloaders( dataset_name, args ):
    test_data = get_dataset( dataset_name, args, train=False )

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=8 )

    return test_queue

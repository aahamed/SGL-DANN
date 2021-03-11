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
    else:
        assert False and f'Unrecognized dataset: {dataset_name}'
    return train_data
    
def get_dataloaders( dataset_name, args ):
    train_data = get_train_dataset( dataset_name, args )
    num_train = len( train_data )
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=4 )

    unlabeled_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=4 )

    valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(
          indices[split:num_train]),
      pin_memory=True, num_workers=4 )

    return train_queue, unlabeled_queue, valid_queue


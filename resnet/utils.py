""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_network(args, num_class):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_class=num_class)
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(num_class=num_class)
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_class=num_class)
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn(num_class=num_class)
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121(num_class=num_class)
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161(num_class=num_class)
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169(num_class=num_class)
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201(num_class=num_class)
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet(num_class=num_class)
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3(num_class=num_class)
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4(num_class=num_class)
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2(num_class=num_class)
    elif args.net == 'xception':
        from models.xception import xception
        net = xception(num_class=num_class)
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_class=num_class)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_class=num_class)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_class=num_class)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_class=num_class)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(num_class=num_class)
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18(num_class=num_class)
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34(num_class=num_class)
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50(num_class=num_class)
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101(num_class=num_class)
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152(num_class=num_class)
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50(num_class=num_class)
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101(num_class=num_class)
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152(num_class=num_class)
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet(num_class=num_class)
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2(num_class=num_class)
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet(num_class)
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet(class_num=num_class)
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(num_class=num_class)
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet(num_class=num_class)
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56(num_class=num_class)
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92(num_class=num_class)
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18(num_class=num_class)
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34(num_class=num_class)
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50(num_class=num_class)
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101(num_class=num_class)
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152(num_class=num_class)
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet(num_class=num_class)
    # elif args.net == 'resnetnew20':
    #     from models.resnet_new import resnet20
    #     net = resnet20(num_class=num_class)
    # elif args.net == 'resnetnew32':
    #     from models.resnet_new import resnet32
    #     net = resnet32(num_class=num_class)
    # elif args.net == 'resnetnew44':
    #     from models.resnet_new import resnet44
    #     net = resnet44(num_class=num_class)
    # elif args.net == 'resnetnew56':
    #     from models.resnet_new import resnet56
    #     net = resnet56(num_class=num_class)
    # elif args.net == 'resnetnew110':
    #     from models.resnet_new import resnet110
    #     net = resnet110(num_class=num_class)
    elif args.net == 'resnet110new':
        from models.resnewnew import resnet110_btnk
        net = resnet110_btnk(num_classes=num_class)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:  # use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, dataset_name='cifar100'):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
        dataset_name: the name of dataset to use
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    if dataset_name == 'cifar100':
        cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                          transform=transform_train)
        cifar100_training_loader = DataLoader(cifar100_training, shuffle=shuffle, num_workers=num_workers,
                                              batch_size=batch_size)
        return cifar100_training_loader
    elif dataset_name == 'cifar10':
        cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                        transform=transform_train)
        cifar10_training_loader = DataLoader(cifar10_training, shuffle=shuffle, num_workers=num_workers,
                                             batch_size=batch_size)
        return cifar10_training_loader
    elif dataset_name == 'svhn':
        svhn_training = torchvision.datasets.SVHN(root='./data', split='train', download=True,
                                                  transform=transform_train)
        svhn_training_loader = DataLoader(svhn_training, shuffle=shuffle, num_workers=num_workers,
                                          batch_size=batch_size)
        return svhn_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, dataset_name='cifar100'):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
        dataset_name: the name of dataset to use
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    if dataset_name == 'cifar100':
        cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                      transform=transform_test)
        cifar100_test_loader = DataLoader(cifar100_test, shuffle=shuffle, num_workers=num_workers,
                                          batch_size=batch_size)
        return cifar100_test_loader
    elif dataset_name == 'cifar10':
        cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        cifar10_test_loader = DataLoader(cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return cifar10_test_loader
    elif dataset_name == 'svhn':
        svhn_test = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
        svhn_test_loader = DataLoader(svhn_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return svhn_test_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
        raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch


def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

# train.py
# !/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from frgd.sgd_stiefel import SGDStiefel


def train(epoch):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(data_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(data_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(data_training_loader.dataset)
        ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in data_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(data_test_loader.dataset),
        correct.float() / len(data_test_loader.dataset),
        finish - start
    ))
    print()

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(data_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(data_test_loader.dataset), epoch)

    return correct.float() / len(data_test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-log', type=str, default='', help='additional information in the name of log file')
    parser.add_argument('-ds', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'svhn'],
                        help='dataset name')
    # FRGD
    parser.add_argument('-frgd', action='store_true', default=False, help='use FRGD')
    parser.add_argument('-lrg', type=float, default=0.1, help='initial learning rate for FRGD')
    parser.add_argument('-stiefel', type=float, default=0., help='lr ratio for stiefel')
    parser.add_argument('-oblique', type=float, default=0., help='lr ratio for oblique')
    parser.add_argument('-feedback', type=float, default=0., help='feedback parameter in FRGD')
    args = parser.parse_args()

    # data preprocessing:
    if args.ds == 'cifar100':
        num_class = 100
        mean = settings.CIFAR100_TRAIN_MEAN
        std = settings.CIFAR100_TRAIN_STD
    elif args.ds == 'cifar10':
        num_class = 10
        mean = settings.CIFAR10_TRAIN_MEAN
        std = settings.CIFAR10_TRAIN_STD
    elif args.ds == 'svhn':
        num_class = 10
        mean = settings.SVHN_TRAIN_MEAN
        std = settings.SVHN_TRAIN_STD
    # elif args.dataset == 'imagenet':
    #     num_classes = 1000
    #     mean = settings.IMAGENET_TRAIN_MEAN
    #     std = settings.IMAGENET_TRAIN_STD
    else:  # dataset name ERROR
        print('the dataset name you have entered is not supported yet')
        sys.exit()

    net = get_network(args, num_class)
    net = nn.DataParallel(net)

    #data preprocessing:
    data_training_loader = get_training_dataloader(
        mean,
        std,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        dataset_name=args.ds
    )

    data_test_loader = get_test_dataloader(
        mean,
        std,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        dataset_name=args.ds
    )
    
    def qr_retraction(tan_vec):  # tan_vec, p-by-n, p <= n
        tan_vec.t_()
        q, r = torch.qr(tan_vec)
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph.expand_as(q)
        q.t_()
        return q


    loss_function = nn.CrossEntropyLoss()
    if not args.frgd:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        param_g = []
        param_e = []

        for p in net.parameters():
            if p.dim() == 4 and p.size()[0] <= np.prod(p.size()[1:]):  # stiefel
                q = qr_retraction(p.data.view(p.size(0), -1))
                p.data.copy_(q.view(p.size()))
                param_g.append(p)
            else:
                param_e.append(p)
        dict_g = {'params': param_g, 'lr': args.lrg, 'momentum': 0.9, 'weight_decay': 5e-4,
                  'stiefel': args.stiefel, 'oblique': args.oblique, 'feedback': (args.feedback, args.feedback)}
        dict_e = {'params': param_e, 'lr': args.lr, 'momentum': 0.9, 'weight_decay': 5e-4,
                  'stiefel': 0, 'oblique': 0, 'feedback': (0, 0)}
        optimizer = SGDStiefel([dict_g, dict_e])

    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    iter_per_epoch = len(data_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # since tensorboard can't overwrite old values
    # so the only way is to create a new tensorboard log
    log_name = settings.TIME_NOW if not args.log else args.log + '_' + settings.TIME_NOW
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, log_name))
    input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
    writer.add_graph(net.module, input_tensor)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()

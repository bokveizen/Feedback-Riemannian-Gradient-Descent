""" train network using pytorch
author baiyu
modified by Fanchen Bu
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from dataset import *
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

from sgd_original import SGD
from sgd_oblique import SGDOblique
from sgd_stiefel import SGDStiefel
from sgd_oblique_decomp import SGDObliqueDecomp
from sgd_stiefel_decomp import SGDStiefelDecomp
from sgd_oblique_wn2 import SGDObliqueWN
from sgd_stiefel_wn2 import SGDStiefelWN
from sgd_oblique_kk import SGDObliqueKK
from sgd_stiefel_kk import SGDStiefelKK
from sgd_oblique_cc import SGDObliqueCC
from sgd_stiefel_cc import SGDStiefelCC

from scaled_for_pruning import default_threshold_rate


def train(epoch, use_gpu):
    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)
        if use_gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1
        cur_index_fc = 0
        cur_index_conv = 0
        for p in net.parameters():
            if p.dim() == 2:  # FC weight
                norm_mean = p.data.norm(dim=1).mean()
                norm_var = p.data.norm(dim=1).var()
                dist_from_st = (p.data.mm(p.data.t()) - torch.eye(p.data.shape[0], device=p.device)).norm()
                writer.add_scalar('LinearLayer{}WeightsNorms/norm_mean'.format(cur_index_fc),
                                  norm_mean, n_iter)
                writer.add_scalar('LinearLayer{}WeightsNorms/norm_var'.format(cur_index_fc),
                                  norm_var, n_iter)
                writer.add_scalar('LinearLayer{}WeightsNorms/dist_from_st'.format(cur_index_fc),
                                  dist_from_st, n_iter)
                cur_index_fc += 1
            elif p.dim() == 4:  # Conv weight
                p_2d = p.data.view(p.data.shape[0], -1)
                norm_mean = p_2d.norm(dim=1).mean()
                norm_var = p_2d.norm(dim=1).var()
                dist_from_st = (p_2d.mm(p_2d.t()) - torch.eye(p_2d.shape[0], device=p.device)).norm()
                writer.add_scalar('ConvLayer{}WeightsNorms/norm_mean'.format(cur_index_conv),
                                  norm_mean, n_iter)
                writer.add_scalar('ConvLayer{}WeightsNorms/norm_var'.format(cur_index_conv),
                                  norm_var, n_iter)
                writer.add_scalar('ConvLayer{}WeightsNorms/dist_from_st'.format(cur_index_conv),
                                  dist_from_st, n_iter)
                cur_index_conv += 1
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
            trained_samples=batch_index * args.batchsize + len(images),
            total_samples=len(training_loader.dataset)
        ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)


def eval_training(epoch, use_gpu):
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in test_loader:
        images = Variable(images)
        labels = Variable(labels)
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset)
    ))
    print()

    # add information to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--net', type=str, required=True, help='the net type to use')
    parser.add_argument('-g', '--gpu', type=bool, default=torch.cuda.is_available(), help='use gpu or not')
    parser.add_argument('-ds', '--dataset', type=str, default='imagenet', help='the name of dataset')
    parser.add_argument('-nw', '--numworkers', type=int, default=0, help='the number of workers for dataloader')
    parser.add_argument('-b', '--batchsize', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-shf', '--shuffle', type=bool, default=True, help='shuffle the dataset or not')
    parser.add_argument('-w', '--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-l', '--learningrate', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-t', '--thresholdrate', type=float, default=default_threshold_rate, help='threshold rate used for pruning')
    parser.add_argument('-stp', '--stiefelpunishment', type=float, default=0., help='Punishment factor used for stiefel manifold')
    parser.add_argument('-d', '--decomposition', type=str, default='', help='decomposition type used for conv weights decomposition')
    mani_group = parser.add_mutually_exclusive_group()
    mani_group.add_argument('-o', '--oblique', action='store_true', help='use optimizer on oblique manifold')
    mani_group.add_argument('-s', '--stiefel', action='store_true', help='use optimizer on stiefel manifold')
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument('-p', '--parainitob', action='store_true', help='weight initialization on oblique manifold')
    init_group.add_argument('-P', '--parainitst', action='store_true', help='weight initialization on stiefel manifold')
    optim_group = parser.add_mutually_exclusive_group()
    optim_group.add_argument('-k', '--kernel', action='store_true', help='apply feedback GD on kernel matrix')
    optim_group.add_argument('-c', '--channel', action='store_true', help='apply feedback GD on channel matrix')
    parser.add_argument('-f', '--feedback', action='store_true', help='use feedback integrator')
    args = parser.parse_args()

    if args.dataset == 'cifar100':
        num_classes = 100
        mean = settings.CIFAR100_TRAIN_MEAN
        std = settings.CIFAR100_TRAIN_STD
    elif args.dataset == 'cifar10':
        num_classes = 10
        mean = settings.CIFAR10_TRAIN_MEAN
        std = settings.CIFAR10_TRAIN_STD
    elif args.dataset == 'svhn':
        num_classes = 10
        mean = settings.SVHN_TRAIN_MEAN
        std = settings.SVHN_TRAIN_STD
    elif args.dataset == 'imagenet':
        num_classes = 1000
        mean = settings.IMAGENET_TRAIN_MEAN
        std = settings.IMAGENET_TRAIN_STD
    else:  # dataset name ERROR
        # num_classes = None
        # mean = None
        # std = None
        print('the dataset name you have entered is not supported yet')
        sys.exit()
    net = get_network(args, use_gpu=args.gpu, num_classes=num_classes)

    # oblique parainit
    if args.oblique or args.parainitob:
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=1.)
                m.weight.data.div_(m.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.
                # nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight'):  # conv decomposition will delete the attribution 'weight'
                    if args.kernel:
                        v = m.weight.data
                        v_f = v.view(-1, v.shape[2], v.shape[3])
                        for f in v_f:  # f has shape (k, k) and is on Oblique manifold
                            nn.init.normal_(f)
                            f.div_(f.norm(dim=1).view(-1, 1))
                    elif args.channel:
                        v = m.weight.data
                        v_f = v.permute(2, 3, 0, 1).view(-1, v.shape[0], v.shape[1])
                        for f in v_f:  # f has shape (c_out, c_in) and is on Oblique manifold
                            nn.init.normal_(f)
                            f.div_(f.norm(dim=1).view(-1, 1))
                    else:
                        nn.init.normal_(m.weight, mean=0., std=1.)
                        m.weight.data.div_(m.weight.data.view(m.weight.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1))  # one-line ver.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    # stiefel parainit
    elif args.stiefel or args.parainitst:
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                m.weight.data.div_(m.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight'):  # conv decomposition will delete the attribution 'weight'
                    if args.kernel:
                        v = m.weight.data
                        v_f = v.view(-1, v.shape[2], v.shape[3])
                        for f in v_f:  # f has shape (k, k) and is on Oblique manifold
                            nn.init.orthogonal_(f)
                            f.div_(f.norm(dim=1).view(-1, 1))
                    elif args.channel:
                        v = m.weight.data
                        v_f = v.permute(2, 3, 0, 1).view(-1, v.shape[0], v.shape[1])
                        for f in v_f:  # f has shape (c_out, c_in) and is on Oblique manifold
                            nn.init.orthogonal_(f)
                            f.div_(f.norm(dim=1).view(-1, 1))
                    else:
                        nn.init.orthogonal_(m.weight)
                        m.weight.data.div_(m.weight.data.view(m.weight.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1))  # one-line ver.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    # data preprocessing:
    training_loader = get_training_dataloader(
        mean, std,
        num_workers=args.numworkers,
        batch_size=args.batchsize,
        shuffle=args.shuffle,
        dataset_name=args.dataset
    )

    test_loader = get_test_dataloader(
        mean, std,
        num_workers=args.numworkers,
        batch_size=args.batchsize,
        shuffle=args.shuffle,
        dataset_name=args.dataset
    )

    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4)
    if args.oblique:
        if args.decomposition:
            optimizer = SGDObliqueDecomp(net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4, feedback=args.feedback, decomp_type=args.decomposition)
        elif 'wn2' in args.net:
            optimizer = SGDObliqueWN(net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4, feedback=args.feedback)
        elif args.kernel:
            optimizer = SGDObliqueKK(net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4, feedback=args.feedback)
        elif args.channel:
            optimizer = SGDObliqueCC(net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4, feedback=args.feedback)
        else:
            optimizer = SGDOblique(net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4, feedback=args.feedback)
    elif args.stiefel:
        if args.decomposition:
            optimizer = SGDStiefelDecomp(net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4,
                                         feedback=args.feedback, punishment=args.stiefelpunishment, decomp_type=args.decomposition)
        elif 'wn2' in args.net:
            optimizer = SGDStiefelWN(net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4, feedback=args.feedback)
        elif args.kernel:
            optimizer = SGDStiefelKK(net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4, feedback=args.feedback)
        elif args.channel:
            optimizer = SGDStiefelCC(net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4, feedback=args.feedback)
        else:
            optimizer = SGDStiefel(net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4,
                                   feedback=args.feedback, punishment=args.stiefelpunishment)
    else:
        optimizer = SGD(net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.5)  # learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    tag = args.dataset + '_'
    if args.parainitob:
        tag += 'p'
    if args.parainitst:
        tag += 'P'
    if args.oblique:
        tag += 'o'
    if args.stiefel:
        tag += 's'
    if args.feedback:
        tag += 'f'
    if args.kernel:
        tag += 'k'
    if args.channel:
        tag += 'c'
    if 'p' in args.net:
        tag += '_t_' + str(args.thresholdrate)
    if args.stiefelpunishment > 0:
        tag += '_stp_' + str(args.stiefelpunishment)
    if args.decomposition:
        tag += '_decomp_' + args.decomposition
    log_file_name = tag + '_' + settings.TIME_NOW
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, log_file_name)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, log_file_name))
    input_tensor = torch.Tensor(12, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch, args.gpu)
        acc = eval_training(epoch, args.gpu)
        net_tag = args.net + tag
        # start to save best performance model after MILESTONE # 1
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=net_tag, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=net_tag, epoch=epoch, type='regular'))

    writer.close()

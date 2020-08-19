""" helper function

author baiyu
"""

import sys
import numpy
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# from dataset import CIFAR100Train, CIFAR100Test

def get_network(args, use_gpu=True, num_classes=100):
    """ return given network
    """
    # resnet
    if args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_classes)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(num_classes)

    # resnet_wn
    elif args.net == 'resnet18wn':
        from models.resnet_wn import resnet18
        net = resnet18(num_classes)
    elif args.net == 'resnet34wn':
        from models.resnet_wn import resnet34
        net = resnet34(num_classes)
    elif args.net == 'resnet50wn':
        from models.resnet_wn import resnet50
        net = resnet50(num_classes)
    elif args.net == 'resnet101wn':
        from models.resnet_wn import resnet101
        net = resnet101(num_classes)
    elif args.net == 'resnet152wn':
        from models.resnet_wn import resnet152
        net = resnet152(num_classes)

    # resnet_wn_channel_wise
    elif args.net == 'resnet18wncw':
        from models.resnet_wn_channel_wise import resnet18
        net = resnet18(num_classes)
    elif args.net == 'resnet34wncw':
        from models.resnet_wn_channel_wise import resnet34
        net = resnet34(num_classes)
    elif args.net == 'resnet50wncw':
        from models.resnet_wn_channel_wise import resnet50
        net = resnet50(num_classes)
    elif args.net == 'resnet101wncw':
        from models.resnet_wn_channel_wise import resnet101
        net = resnet101(num_classes)
    elif args.net == 'resnet152wncw':
        from models.resnet_wn_channel_wise import resnet152
        net = resnet152(num_classes)

    # resnet_wn_conv_only
    elif args.net == 'resnet18wnc':
        from models.resnet_wn_conv_only import resnet18
        net = resnet18(num_classes)
    elif args.net == 'resnet34wnc':
        from models.resnet_wn_conv_only import resnet34
        net = resnet34(num_classes)
    elif args.net == 'resnet50wnc':
        from models.resnet_wn_conv_only import resnet50
        net = resnet50(num_classes)
    elif args.net == 'resnet101wnc':
        from models.resnet_wn_conv_only import resnet101
        net = resnet101(num_classes)
    elif args.net == 'resnet152wnc':
        from models.resnet_wn_conv_only import resnet152
        net = resnet152(num_classes)

    # resnet_wn_conv_only_channel_wise
    elif args.net == 'resnet18wnccw':
        from models.resnet_wn_conv_only_channel_wise import resnet18
        net = resnet18(num_classes)
    elif args.net == 'resnet34wnccw':
        from models.resnet_wn_conv_only_channel_wise import resnet34
        net = resnet34(num_classes)
    elif args.net == 'resnet50wnccw':
        from models.resnet_wn_conv_only_channel_wise import resnet50
        net = resnet50(num_classes)
    elif args.net == 'resnet101wnccw':
        from models.resnet_wn_conv_only_channel_wise import resnet101
        net = resnet101(num_classes)
    elif args.net == 'resnet152wnccw':
        from models.resnet_wn_conv_only_channel_wise import resnet152
        net = resnet152(num_classes)

    # resnet_scaled
    elif args.net == 'resnet18s':
        from models.resnet_scaled import resnet18
        net = resnet18(args, num_classes)
    elif args.net == 'resnet34s':
        from models.resnet_scaled import resnet34
        net = resnet34(args, num_classes)
    elif args.net == 'resnet50s':
        from models.resnet_scaled import resnet50
        net = resnet50(args, num_classes)
    elif args.net == 'resnet101s':
        from models.resnet_scaled import resnet101
        net = resnet101(args, num_classes)
    elif args.net == 'resnet152s':
        from models.resnet_scaled import resnet152
        net = resnet152(args, num_classes)

    # resnet_scaled_pruned
    elif args.net == 'resnet18sp':
        from models.resnet_scaled_pruned import resnet18
        net = resnet18(args, num_classes)
    elif args.net == 'resnet34sp':
        from models.resnet_scaled_pruned import resnet34
        net = resnet34(args, num_classes)
    elif args.net == 'resnet50sp':
        from models.resnet_scaled_pruned import resnet50
        net = resnet50(args, num_classes)
    elif args.net == 'resnet101sp':
        from models.resnet_scaled_pruned import resnet101
        net = resnet101(args, num_classes)
    elif args.net == 'resnet152sp':
        from models.resnet_scaled_pruned import resnet152
        net = resnet152(args, num_classes)

    # resnet_no_bn
    elif args.net == 'resnet18nb':
        from models.resnet_no_bn import resnet18
        net = resnet18(num_classes)
    elif args.net == 'resnet34nb':
        from models.resnet_no_bn import resnet34
        net = resnet34(num_classes)
    elif args.net == 'resnet50nb':
        from models.resnet_no_bn import resnet50
        net = resnet50(num_classes)
    elif args.net == 'resnet101nb':
        from models.resnet_no_bn import resnet101
        net = resnet101(num_classes)
    elif args.net == 'resnet152nb':
        from models.resnet_no_bn import resnet152
        net = resnet152(num_classes)

    # resnet_no_bn_wn
    elif args.net == 'resnet18nbwn':
        from models.resnet_no_bn_wn import resnet18
        net = resnet18(num_classes)
    elif args.net == 'resnet34nbwn':
        from models.resnet_no_bn_wn import resnet34
        net = resnet34(num_classes)
    elif args.net == 'resnet50nbwn':
        from models.resnet_no_bn_wn import resnet50
        net = resnet50(num_classes)
    elif args.net == 'resnet101nbwn':
        from models.resnet_no_bn_wn import resnet101
        net = resnet101(num_classes)
    elif args.net == 'resnet152nbwn':
        from models.resnet_no_bn_wn import resnet152
        net = resnet152(num_classes)

    # resnet_no_bn_wn_conv_only
    elif args.net == 'resnet18nbwnc':
        from models.resnet_no_bn_wn_conv_only import resnet18
        net = resnet18(num_classes)
    elif args.net == 'resnet34nbwnc':
        from models.resnet_no_bn_wn_conv_only import resnet34
        net = resnet34(num_classes)
    elif args.net == 'resnet50nbwnc':
        from models.resnet_no_bn_wn_conv_only import resnet50
        net = resnet50(num_classes)
    elif args.net == 'resnet101nbwnc':
        from models.resnet_no_bn_wn_conv_only import resnet101
        net = resnet101(num_classes)
    elif args.net == 'resnet152nbwnc':
        from models.resnet_no_bn_wn_conv_only import resnet152
        net = resnet152(num_classes)

    # resnet_no_bn_wn_channel_wise
    elif args.net == 'resnet18nbwncw':
        from models.resnet_no_bn_wn_channel_wise import resnet18
        net = resnet18(num_classes)
    elif args.net == 'resnet34nbwncw':
        from models.resnet_no_bn_wn_channel_wise import resnet34
        net = resnet34(num_classes)
    elif args.net == 'resnet50nbwncw':
        from models.resnet_no_bn_wn_channel_wise import resnet50
        net = resnet50(num_classes)
    elif args.net == 'resnet101nbwncw':
        from models.resnet_no_bn_wn_channel_wise import resnet101
        net = resnet101(num_classes)
    elif args.net == 'resnet152nbwncw':
        from models.resnet_no_bn_wn_channel_wise import resnet152
        net = resnet152(num_classes)

    # resnet_no_bn_wn_conv_only_channel_wise
    elif args.net == 'resnet18nbwnccw':
        from models.resnet_no_bn_wn_conv_only_channel_wise import resnet18
        net = resnet18(num_classes)
    elif args.net == 'resnet34nbwnccw':
        from models.resnet_no_bn_wn_conv_only_channel_wise import resnet34
        net = resnet34(num_classes)
    elif args.net == 'resnet50nbwnccw':
        from models.resnet_no_bn_wn_conv_only_channel_wise import resnet50
        net = resnet50(num_classes)
    elif args.net == 'resnet101nbwnccw':
        from models.resnet_no_bn_wn_conv_only_channel_wise import resnet101
        net = resnet101(num_classes)
    elif args.net == 'resnet152nbwnccw':
        from models.resnet_no_bn_wn_conv_only_channel_wise import resnet152
        net = resnet152(num_classes)

    # resnet_no_bn_scaled
    elif args.net == 'resnet18nbs':
        from models.resnet_no_bn_scaled import resnet18
        net = resnet18(args, num_classes)
    elif args.net == 'resnet34nbs':
        from models.resnet_no_bn_scaled import resnet34
        net = resnet34(args, num_classes)
    elif args.net == 'resnet50nbs':
        from models.resnet_no_bn_scaled import resnet50
        net = resnet50(args, num_classes)
    elif args.net == 'resnet101nbs':
        from models.resnet_no_bn_scaled import resnet101
        net = resnet101(args, num_classes)
    elif args.net == 'resnet152nbs':
        from models.resnet_no_bn_scaled import resnet152
        net = resnet152(args, num_classes)

    # resnet_no_bn_scaled_pruned
    elif args.net == 'resnet18nbsp':
        from models.resnet_no_bn_scaled_pruned import resnet18
        net = resnet18(args, num_classes)
    elif args.net == 'resnet34nbsp':
        from models.resnet_no_bn_scaled_pruned import resnet34
        net = resnet34(args, num_classes)
    elif args.net == 'resnet50nbsp':
        from models.resnet_no_bn_scaled_pruned import resnet50
        net = resnet50(args, num_classes)
    elif args.net == 'resnet101nbsp':
        from models.resnet_no_bn_scaled_pruned import resnet101
        net = resnet101(args, num_classes)
    elif args.net == 'resnet152nbsp':
        from models.resnet_no_bn_scaled_pruned import resnet152
        net = resnet152(args, num_classes)

    # resnet_decomp
    elif args.net == 'resnet18d':
        from models.resnet_decomp import resnet18
        net = resnet18(args, num_classes)
    elif args.net == 'resnet34d':
        from models.resnet_decomp import resnet34
        net = resnet34(args, num_classes)
    elif args.net == 'resnet50d':
        from models.resnet_decomp import resnet50
        net = resnet50(args, num_classes)
    elif args.net == 'resnet101d':
        from models.resnet_decomp import resnet101
        net = resnet101(args, num_classes)
    elif args.net == 'resnet152d':
        from models.resnet_decomp import resnet152
        net = resnet152(args, num_classes)

    # resnet_scaled_decomp
    elif args.net == 'resnet18sd':
        from models.resnet_scaled_decomp import resnet18
        net = resnet18(args, num_classes)
    elif args.net == 'resnet34sd':
        from models.resnet_scaled_decomp import resnet34
        net = resnet34(args, num_classes)
    elif args.net == 'resnet50sd':
        from models.resnet_scaled_decomp import resnet50
        net = resnet50(args, num_classes)
    elif args.net == 'resnet101sd':
        from models.resnet_scaled_decomp import resnet101
        net = resnet101(args, num_classes)
    elif args.net == 'resnet152sd':
        from models.resnet_scaled_decomp import resnet152
        net = resnet152(args, num_classes)

    # resnet_scaled_pruned_decomp
    elif args.net == 'resnet18spd':
        from models.resnet_scaled_pruned_decomp import resnet18
        net = resnet18(args, num_classes)
    elif args.net == 'resnet34spd':
        from models.resnet_scaled_pruned_decomp import resnet34
        net = resnet34(args, num_classes)
    elif args.net == 'resnet50spd':
        from models.resnet_scaled_pruned_decomp import resnet50
        net = resnet50(args, num_classes)
    elif args.net == 'resnet101spd':
        from models.resnet_scaled_pruned_decomp import resnet101
        net = resnet101(args, num_classes)
    elif args.net == 'resnet152spd':
        from models.resnet_scaled_pruned_decomp import resnet152
        net = resnet152(args, num_classes)

    # resnet_wn_2d
    elif args.net == 'resnet18wn2':
        from models.resnet_wn_2d import resnet18
        net = resnet18(args, num_classes)
    elif args.net == 'resnet34wn2':
        from models.resnet_wn_2d import resnet34
        net = resnet34(args, num_classes)
    elif args.net == 'resnet50wn2':
        from models.resnet_wn_2d import resnet50
        net = resnet50(args, num_classes)
    elif args.net == 'resnet101wn2':
        from models.resnet_wn_2d import resnet101
        net = resnet101(args, num_classes)
    elif args.net == 'resnet152wn2':
        from models.resnet_wn_2d import resnet152
        net = resnet152(args, num_classes)

    ################################## RESNET NO BN ########################################
    # resnet_no_bn_decomp
    elif args.net == 'resnet18nbd':
        from models.resnet_no_bn_decomp import resnet18
        net = resnet18(args, num_classes)
    elif args.net == 'resnet34nbd':
        from models.resnet_no_bn_decomp import resnet34
        net = resnet34(args, num_classes)
    elif args.net == 'resnet50nbd':
        from models.resnet_no_bn_decomp import resnet50
        net = resnet50(args, num_classes)
    elif args.net == 'resnet101nbd':
        from models.resnet_no_bn_decomp import resnet101
        net = resnet101(args, num_classes)
    elif args.net == 'resnet152nbd':
        from models.resnet_no_bn_decomp import resnet152
        net = resnet152(args, num_classes)

    # resnet_no_bn_scaled_decomp
    elif args.net == 'resnet18nbsd':
        from models.resnet_no_bn_scaled_decomp import resnet18
        net = resnet18(args, num_classes)
    elif args.net == 'resnet34nbsd':
        from models.resnet_no_bn_scaled_decomp import resnet34
        net = resnet34(args, num_classes)
    elif args.net == 'resnet50nbsd':
        from models.resnet_no_bn_scaled_decomp import resnet50
        net = resnet50(args, num_classes)
    elif args.net == 'resnet101nbsd':
        from models.resnet_no_bn_scaled_decomp import resnet101
        net = resnet101(args, num_classes)
    elif args.net == 'resnet152nbsd':
        from models.resnet_no_bn_scaled_decomp import resnet152
        net = resnet152(args, num_classes)

    # resnet_no_bn_scaled_pruned_decomp
    elif args.net == 'resnet18nbspd':
        from models.resnet_no_bn_scaled_pruned_decomp import resnet18
        net = resnet18(args, num_classes)
    elif args.net == 'resnet34nbspd':
        from models.resnet_no_bn_scaled_pruned_decomp import resnet34
        net = resnet34(args, num_classes)
    elif args.net == 'resnet50nbspd':
        from models.resnet_no_bn_scaled_pruned_decomp import resnet50
        net = resnet50(args, num_classes)
    elif args.net == 'resnet101nbspd':
        from models.resnet_no_bn_scaled_pruned_decomp import resnet101
        net = resnet101(args, num_classes)
    elif args.net == 'resnet152nbspd':
        from models.resnet_no_bn_scaled_pruned_decomp import resnet152
        net = resnet152(args, num_classes)

    # resnet_no_bn_wn_2d
    elif args.net == 'resnet18nbwn2':
        from models.resnet_no_bn_wn_2d import resnet18
        net = resnet18(args, num_classes)
    elif args.net == 'resnet34nbwn2':
        from models.resnet_no_bn_wn_2d import resnet34
        net = resnet34(args, num_classes)
    elif args.net == 'resnet50nbwn2':
        from models.resnet_no_bn_wn_2d import resnet50
        net = resnet50(args, num_classes)
    elif args.net == 'resnet101nbwn2':
        from models.resnet_no_bn_wn_2d import resnet101
        net = resnet101(args, num_classes)
    elif args.net == 'resnet152nbwn2':
        from models.resnet_no_bn_wn_2d import resnet152
        net = resnet152(args, num_classes)

    # vgg
    elif args.net == 'vgg11':
        from models.vgg import vgg11
        net = vgg11(num_classes)
    elif args.net == 'vgg13':
        from models.vgg import vgg13
        net = vgg13(num_classes)
    elif args.net == 'vgg16':
        from models.vgg import vgg16
        net = vgg16(num_classes)
    elif args.net == 'vgg19':
        from models.vgg import vgg19
        net = vgg19(num_classes)
    elif args.net == 'vgg11bn':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_classes)
    elif args.net == 'vgg13bn':
        from models.vgg import vgg13_bn
        net = vgg13_bn(num_classes)
    elif args.net == 'vgg16bn':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_classes)
    elif args.net == 'vgg19bn':
        from models.vgg import vgg19_bn
        net = vgg19_bn(num_classes)

    # vgg_wn
    elif args.net == 'vgg11wn':
        from models.vgg_wn import vgg11
        net = vgg11(num_classes)
    elif args.net == 'vgg13wn':
        from models.vgg_wn import vgg13
        net = vgg13(num_classes)
    elif args.net == 'vgg16wn':
        from models.vgg_wn import vgg16
        net = vgg16(num_classes)
    elif args.net == 'vgg19wn':
        from models.vgg_wn import vgg19
        net = vgg19(num_classes)
    elif args.net == 'vgg11bnwn':
        from models.vgg_wn import vgg11_bn
        net = vgg11_bn(num_classes)
    elif args.net == 'vgg13bnwn':
        from models.vgg_wn import vgg13_bn
        net = vgg13_bn(num_classes)
    elif args.net == 'vgg16bnwn':
        from models.vgg_wn import vgg16_bn
        net = vgg16_bn(num_classes)
    elif args.net == 'vgg19bnwn':
        from models.vgg_wn import vgg19_bn
        net = vgg19_bn(num_classes)

    # vgg_wn_conv_only
    elif args.net == 'vgg11wnc':
        from models.vgg_wn_conv_only import vgg11
        net = vgg11(num_classes)
    elif args.net == 'vgg13wnc':
        from models.vgg_wn_conv_only import vgg13
        net = vgg13(num_classes)
    elif args.net == 'vgg16wnc':
        from models.vgg_wn_conv_only import vgg16
        net = vgg16(num_classes)
    elif args.net == 'vgg19wnc':
        from models.vgg_wn_conv_only import vgg19
        net = vgg19(num_classes)
    elif args.net == 'vgg11bnwnc':
        from models.vgg_wn_conv_only import vgg11_bn
        net = vgg11_bn(num_classes)
    elif args.net == 'vgg13bnwnc':
        from models.vgg_wn_conv_only import vgg13_bn
        net = vgg13_bn(num_classes)
    elif args.net == 'vgg16bnwnc':
        from models.vgg_wn_conv_only import vgg16_bn
        net = vgg16_bn(num_classes)
    elif args.net == 'vgg19bnwnc':
        from models.vgg_wn_conv_only import vgg19_bn
        net = vgg19_bn(num_classes)

    # vgg_wn_channel_wise
    elif args.net == 'vgg11wncw':
        from models.vgg_wn_channel_wise import vgg11
        net = vgg11(num_classes)
    elif args.net == 'vgg13wncw':
        from models.vgg_wn_channel_wise import vgg13
        net = vgg13(num_classes)
    elif args.net == 'vgg16wncw':
        from models.vgg_wn_channel_wise import vgg16
        net = vgg16(num_classes)
    elif args.net == 'vgg19wncw':
        from models.vgg_wn_channel_wise import vgg19
        net = vgg19(num_classes)
    elif args.net == 'vgg11bnwncw':
        from models.vgg_wn_channel_wise import vgg11_bn
        net = vgg11_bn(num_classes)
    elif args.net == 'vgg13bnwncw':
        from models.vgg_wn_channel_wise import vgg13_bn
        net = vgg13_bn(num_classes)
    elif args.net == 'vgg16bnwncw':
        from models.vgg_wn_channel_wise import vgg16_bn
        net = vgg16_bn(num_classes)
    elif args.net == 'vgg19bnwncw':
        from models.vgg_wn_channel_wise import vgg19_bn
        net = vgg19_bn(num_classes)

    # vgg_wn_conv_only_channel_wise
    elif args.net == 'vgg11wnccw':
        from models.vgg_wn_conv_only_channel_wise import vgg11
        net = vgg11(num_classes)
    elif args.net == 'vgg13wnccw':
        from models.vgg_wn_conv_only_channel_wise import vgg13
        net = vgg13(num_classes)
    elif args.net == 'vgg16wnccw':
        from models.vgg_wn_conv_only_channel_wise import vgg16
        net = vgg16(num_classes)
    elif args.net == 'vgg19wnccw':
        from models.vgg_wn_conv_only_channel_wise import vgg19
        net = vgg19(num_classes)
    elif args.net == 'vgg11bnwnccw':
        from models.vgg_wn_conv_only_channel_wise import vgg11_bn
        net = vgg11_bn(num_classes)
    elif args.net == 'vgg13bnwnccw':
        from models.vgg_wn_conv_only_channel_wise import vgg13_bn
        net = vgg13_bn(num_classes)
    elif args.net == 'vgg16bnwnccw':
        from models.vgg_wn_conv_only_channel_wise import vgg16_bn
        net = vgg16_bn(num_classes)
    elif args.net == 'vgg19bnwnccw':
        from models.vgg_wn_conv_only_channel_wise import vgg19_bn
        net = vgg19_bn(num_classes)

    # vgg_scaled
    elif args.net == 'vgg11s':
        from models.vgg_scaled import vgg11
        net = vgg11(num_classes, args)
    elif args.net == 'vgg13s':
        from models.vgg_scaled import vgg13
        net = vgg13(num_classes, args)
    elif args.net == 'vgg16s':
        from models.vgg_scaled import vgg16
        net = vgg16(num_classes, args)
    elif args.net == 'vgg19s':
        from models.vgg_scaled import vgg19
        net = vgg19(num_classes, args)
    elif args.net == 'vgg11bns':
        from models.vgg_scaled import vgg11_bn
        net = vgg11_bn(num_classes, args)
    elif args.net == 'vgg13bns':
        from models.vgg_scaled import vgg13_bn
        net = vgg13_bn(num_classes, args)
    elif args.net == 'vgg16bns':
        from models.vgg_scaled import vgg16_bn
        net = vgg16_bn(num_classes, args)
    elif args.net == 'vgg19bns':
        from models.vgg_scaled import vgg19_bn
        net = vgg19_bn(num_classes, args)

    # vgg_scaled_pruned
    elif args.net == 'vgg11sp':
        from models.vgg_scaled_pruned import vgg11
        net = vgg11(num_classes, args)
    elif args.net == 'vgg13sp':
        from models.vgg_scaled_pruned import vgg13
        net = vgg13(num_classes, args)
    elif args.net == 'vgg16sp':
        from models.vgg_scaled_pruned import vgg16
        net = vgg16(num_classes, args)
    elif args.net == 'vgg19sp':
        from models.vgg_scaled_pruned import vgg19
        net = vgg19(num_classes, args)
    elif args.net == 'vgg11bnsp':
        from models.vgg_scaled_pruned import vgg11_bn
        net = vgg11_bn(num_classes, args)
    elif args.net == 'vgg13bnsp':
        from models.vgg_scaled_pruned import vgg13_bn
        net = vgg13_bn(num_classes, args)
    elif args.net == 'vgg16bnsp':
        from models.vgg_scaled_pruned import vgg16_bn
        net = vgg16_bn(num_classes, args)
    elif args.net == 'vgg19bnsp':
        from models.vgg_scaled_pruned import vgg19_bn
        net = vgg19_bn(num_classes, args)

    # vgg_small
    elif args.net == 'vgg11sm':
        from models.vgg_small import vgg11
        net = vgg11(num_classes)
    elif args.net == 'vgg13sm':
        from models.vgg_small import vgg13
        net = vgg13(num_classes)
    elif args.net == 'vgg16sm':
        from models.vgg_small import vgg16
        net = vgg16(num_classes)
    elif args.net == 'vgg19sm':
        from models.vgg_small import vgg19
        net = vgg19(num_classes)
    elif args.net == 'vgg11smbn':
        from models.vgg_small import vgg11_bn
        net = vgg11_bn(num_classes)
    elif args.net == 'vgg13smbn':
        from models.vgg_small import vgg13_bn
        net = vgg13_bn(num_classes)
    elif args.net == 'vgg16smbn':
        from models.vgg_small import vgg16_bn
        net = vgg16_bn(num_classes)
    elif args.net == 'vgg19smbn':
        from models.vgg_small import vgg19_bn
        net = vgg19_bn(num_classes)

    # vgg_small_wn
    elif args.net == 'vgg11smwn':
        from models.vgg_small_wn import vgg11
        net = vgg11(num_classes)
    elif args.net == 'vgg13smwn':
        from models.vgg_small_wn import vgg13
        net = vgg13(num_classes)
    elif args.net == 'vgg16smwn':
        from models.vgg_small_wn import vgg16
        net = vgg16(num_classes)
    elif args.net == 'vgg19smwn':
        from models.vgg_small_wn import vgg19
        net = vgg19(num_classes)
    elif args.net == 'vgg11smbnwn':
        from models.vgg_small_wn import vgg11_bn
        net = vgg11_bn(num_classes)
    elif args.net == 'vgg13smbnwn':
        from models.vgg_small_wn import vgg13_bn
        net = vgg13_bn(num_classes)
    elif args.net == 'vgg16smbnwn':
        from models.vgg_small_wn import vgg16_bn
        net = vgg16_bn(num_classes)
    elif args.net == 'vgg19smbnwn':
        from models.vgg_small_wn import vgg19_bn
        net = vgg19_bn(num_classes)

    # vgg_small_wn_channel_wise
    elif args.net == 'vgg11smwncw':
        from models.vgg_small_wn_channel_wise import vgg11
        net = vgg11(num_classes)
    elif args.net == 'vgg13smwncw':
        from models.vgg_small_wn_channel_wise import vgg13
        net = vgg13(num_classes)
    elif args.net == 'vgg16smwncw':
        from models.vgg_small_wn_channel_wise import vgg16
        net = vgg16(num_classes)
    elif args.net == 'vgg19smwncw':
        from models.vgg_small_wn_channel_wise import vgg19
        net = vgg19(num_classes)
    elif args.net == 'vgg11smbnwncw':
        from models.vgg_small_wn_channel_wise import vgg11_bn
        net = vgg11_bn(num_classes)
    elif args.net == 'vgg13smbnwncw':
        from models.vgg_small_wn_channel_wise import vgg13_bn
        net = vgg13_bn(num_classes)
    elif args.net == 'vgg16smbnwncw':
        from models.vgg_small_wn_channel_wise import vgg16_bn
        net = vgg16_bn(num_classes)
    elif args.net == 'vgg19smbnwncw':
        from models.vgg_small_wn_channel_wise import vgg19_bn
        net = vgg19_bn(num_classes)

    # vgg_small_scaled
    elif args.net == 'vgg11sms':
        from models.vgg_small_scaled import vgg11
        net = vgg11(num_classes, args)
    elif args.net == 'vgg13sms':
        from models.vgg_small_scaled import vgg13
        net = vgg13(num_classes, args)
    elif args.net == 'vgg16sms':
        from models.vgg_small_scaled import vgg16
        net = vgg16(num_classes, args)
    elif args.net == 'vgg19sms':
        from models.vgg_small_scaled import vgg19
        net = vgg19(num_classes, args)
    elif args.net == 'vgg11smbns':
        from models.vgg_small_scaled import vgg11_bn
        net = vgg11_bn(num_classes, args)
    elif args.net == 'vgg13smbns':
        from models.vgg_small_scaled import vgg13_bn
        net = vgg13_bn(num_classes, args)
    elif args.net == 'vgg16smbns':
        from models.vgg_small_scaled import vgg16_bn
        net = vgg16_bn(num_classes, args)
    elif args.net == 'vgg19smbns':
        from models.vgg_small_scaled import vgg19_bn
        net = vgg19_bn(num_classes, args)

    # vgg_small_scaled_pruned
    elif args.net == 'vgg11smsp':
        from models.vgg_small_scaled_pruned import vgg11
        net = vgg11(num_classes, args)
    elif args.net == 'vgg13smsp':
        from models.vgg_small_scaled_pruned import vgg13
        net = vgg13(num_classes, args)
    elif args.net == 'vgg16smsp':
        from models.vgg_small_scaled_pruned import vgg16
        net = vgg16(num_classes, args)
    elif args.net == 'vgg19smsp':
        from models.vgg_small_scaled_pruned import vgg19
        net = vgg19(num_classes, args)
    elif args.net == 'vgg11smbnsp':
        from models.vgg_small_scaled_pruned import vgg11_bn
        net = vgg11_bn(num_classes, args)
    elif args.net == 'vgg13smbnsp':
        from models.vgg_small_scaled_pruned import vgg13_bn
        net = vgg13_bn(num_classes, args)
    elif args.net == 'vgg16smbnsp':
        from models.vgg_small_scaled_pruned import vgg16_bn
        net = vgg16_bn(num_classes, args)
    elif args.net == 'vgg19smbnsp':
        from models.vgg_small_scaled_pruned import vgg19_bn
        net = vgg19_bn(num_classes, args)

    # vgg_small_wn_2d
    elif args.net == 'vgg11smwn2':
        from models.vgg_small_wn_2d import vgg11
        net = vgg11(num_classes, args)
    elif args.net == 'vgg13smwn2':
        from models.vgg_small_wn_2d import vgg13
        net = vgg13(num_classes, args)
    elif args.net == 'vgg16smwn2':
        from models.vgg_small_wn_2d import vgg16
        net = vgg16(num_classes, args)
    elif args.net == 'vgg19smwn2':
        from models.vgg_small_wn_2d import vgg19
        net = vgg19(num_classes, args)
    elif args.net == 'vgg11smbnwn2':
        from models.vgg_small_wn_2d import vgg11_bn
        net = vgg11_bn(num_classes, args)
    elif args.net == 'vgg13smbnwn2':
        from models.vgg_small_wn_2d import vgg13_bn
        net = vgg13_bn(num_classes, args)
    elif args.net == 'vgg16smbnwn2':
        from models.vgg_small_wn_2d import vgg16_bn
        net = vgg16_bn(num_classes, args)
    elif args.net == 'vgg19smbnwn2':
        from models.vgg_small_wn_2d import vgg19_bn
        net = vgg19_bn(num_classes, args)

    # vgg_small_wn_2d_scaled
    elif args.net == 'vgg11smwn2s':
        from models.vgg_small_wn_2d_scaled import vgg11
        net = vgg11(num_classes, args)
    elif args.net == 'vgg13smwn2s':
        from models.vgg_small_wn_2d_scaled import vgg13
        net = vgg13(num_classes, args)
    elif args.net == 'vgg16smwn2s':
        from models.vgg_small_wn_2d_scaled import vgg16
        net = vgg16(num_classes, args)
    elif args.net == 'vgg19smwn2s':
        from models.vgg_small_wn_2d_scaled import vgg19
        net = vgg19(num_classes, args)
    elif args.net == 'vgg11smbnwn2s':
        from models.vgg_small_wn_2d_scaled import vgg11_bn
        net = vgg11_bn(num_classes, args)
    elif args.net == 'vgg13smbnwn2s':
        from models.vgg_small_wn_2d_scaled import vgg13_bn
        net = vgg13_bn(num_classes, args)
    elif args.net == 'vgg16smbnwn2s':
        from models.vgg_small_wn_2d_scaled import vgg16_bn
        net = vgg16_bn(num_classes, args)
    elif args.net == 'vgg19smbnwn2s':
        from models.vgg_small_wn_2d_scaled import vgg19_bn
        net = vgg19_bn(num_classes, args)

    # vgg_small_wn_2d_scaled_pruned
    elif args.net == 'vgg11smwn2sp':
        from models.vgg_small_wn_2d_scaled_pruned import vgg11
        net = vgg11(num_classes, args)
    elif args.net == 'vgg13smwn2sp':
        from models.vgg_small_wn_2d_scaled_pruned import vgg13
        net = vgg13(num_classes, args)
    elif args.net == 'vgg16smwn2sp':
        from models.vgg_small_wn_2d_scaled_pruned import vgg16
        net = vgg16(num_classes, args)
    elif args.net == 'vgg19smwn2sp':
        from models.vgg_small_wn_2d_scaled_pruned import vgg19
        net = vgg19(num_classes, args)
    elif args.net == 'vgg11smbnwn2sp':
        from models.vgg_small_wn_2d_scaled_pruned import vgg11_bn
        net = vgg11_bn(num_classes, args)
    elif args.net == 'vgg13smbnwn2sp':
        from models.vgg_small_wn_2d_scaled_pruned import vgg13_bn
        net = vgg13_bn(num_classes, args)
    elif args.net == 'vgg16smbnwn2sp':
        from models.vgg_small_wn_2d_scaled_pruned import vgg16_bn
        net = vgg16_bn(num_classes, args)
    elif args.net == 'vgg19smbnwn2sp':
        from models.vgg_small_wn_2d_scaled_pruned import vgg19_bn
        net = vgg19_bn(num_classes, args)

    elif args.net == 'wres2810':
        from models.wideresnet import wideresnet2810
        net = wideresnet2810(num_classes)

    # torchvision models
    elif args.net == 'tvresnet18':
        from torchvision.models import resnet18
        net = resnet18(num_classes=num_classes)
    elif args.net == 'tvresnet34':
        from torchvision.models import resnet34
        net = resnet34(num_classes=num_classes)
    elif args.net == 'tvresnet50':
        from torchvision.models import resnet50
        net = resnet50(num_classes=num_classes)
    elif args.net == 'tvresnet101':
        from torchvision.models import resnet101
        net = resnet101(num_classes=num_classes)
    elif args.net == 'tvresnet152':
        from torchvision.models import resnet152
        net = resnet152(num_classes=num_classes)

    elif args.net == 'tvvgg11':
        from torchvision.models import vgg11
        net = vgg11(num_classes=num_classes)
    elif args.net == 'tvvgg13':
        from torchvision.models import vgg13
        net = vgg13(num_classes=num_classes)
    elif args.net == 'tvvgg16':
        from torchvision.models import vgg16
        net = vgg16(num_classes=num_classes)
    elif args.net == 'tvvgg19':
        from torchvision.models import vgg19
        net = vgg19(num_classes=num_classes)
    elif args.net == 'tvvgg11bn':
        from torchvision.models import vgg11_bn
        net = vgg11_bn(num_classes=num_classes)
    elif args.net == 'tvvgg13bn':
        from torchvision.models import vgg13_bn
        net = vgg13_bn(num_classes=num_classes)
    elif args.net == 'tvvgg16bn':
        from torchvision.models import vgg16_bn
        net = vgg16_bn(num_classes=num_classes)
    elif args.net == 'tvvgg19bn':
        from torchvision.models import vgg19_bn
        net = vgg19_bn(num_classes=num_classes)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
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

    if dataset_name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    if dataset_name == 'cifar100':
        cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        cifar100_training_loader = DataLoader(cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return cifar100_training_loader
    elif dataset_name == 'cifar10':
        cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        cifar10_training_loader = DataLoader(cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return cifar10_training_loader
    elif dataset_name == 'svhn':
        svhn_training = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        svhn_training_loader = DataLoader(svhn_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return svhn_training_loader
    elif dataset_name == 'imagenet':
        imagenet_training = torchvision.datasets.ImageFolder('/home/bufhan/imagenet/train', transform=transform_train)
        imagenet_training_loader = DataLoader(imagenet_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True)
        return imagenet_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True, dataset_name='cifar100'):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar100_test_loader:torch dataloader object
    """
    if dataset_name == 'imagenet':
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    if dataset_name == 'cifar100':
        cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        cifar100_test_loader = DataLoader(cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return cifar100_test_loader
    elif dataset_name == 'cifar10':
        cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        cifar10_test_loader = DataLoader(cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return cifar10_test_loader
    elif dataset_name == 'svhn':
        svhn_test = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
        svhn_test_loader = DataLoader(svhn_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        return svhn_test_loader
    elif dataset_name == 'imagenet':
        imagenet_test = torchvision.datasets.ImageFolder('/home/bufhan/imagenet/val', transform=transform_test)
        imagenet_test_loader = DataLoader(imagenet_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True)
        return imagenet_test_loader

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

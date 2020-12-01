"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from scaled import LinearScaled
from scaled_for_pruning import Conv2dScaled, Conv2dScaledWithBN

cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features, num_class=100, args=None):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            LinearScaled(512, num_class, args=args),
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


def make_layers(cfg, batch_norm=False, args=None):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        # layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        #
        # if batch_norm:
        #     layers += [nn.BatchNorm2d(l)]
        if batch_norm:
            layers += [Conv2dScaledWithBN(input_channel, l, kernel_size=3, padding=1, args=args)]
        else:
            layers += [Conv2dScaled(input_channel, l, kernel_size=3, padding=1, args=args)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def vgg11_bn(num_classes=100, args=None):
    return VGG(make_layers(cfg['A'], batch_norm=True, args=args), num_classes, args)


def vgg13_bn(num_classes=100, args=None):
    return VGG(make_layers(cfg['B'], batch_norm=True, args=args), num_classes, args)


def vgg16_bn(num_classes=100, args=None):
    return VGG(make_layers(cfg['D'], batch_norm=True, args=args), num_classes, args)


def vgg19_bn(num_classes=100, args=None):
    return VGG(make_layers(cfg['E'], batch_norm=True, args=args), num_classes, args)


def vgg11(num_classes=100, args=None):
    return VGG(make_layers(cfg['A'], batch_norm=False, args=args), num_classes, args)


def vgg13(num_classes=100, args=None):
    return VGG(make_layers(cfg['B'], batch_norm=False, args=args), num_classes, args)


def vgg16(num_classes=100, args=None):
    return VGG(make_layers(cfg['D'], batch_norm=False, args=args), num_classes, args)


def vgg19(num_classes=100, args=None):
    return VGG(make_layers(cfg['E'], batch_norm=False, args=args), num_classes, args)

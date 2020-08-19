import torch
import torch.nn as nn


class Conv2dScaled(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, args=None):
        super(Conv2dScaled, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.scale = nn.Parameter(torch.ones(out_channels))

        # para init.
        if args is None or args.oblique or args.parainitob:
            nn.init.normal_(self.conv.weight, mean=0., std=1.)
            self.conv.weight.data.div_(
                self.conv.weight.data.view(self.conv.weight.shape[0], -1).norm(dim=1).view(-1, 1, 1,
                                                                                           1))  # one-line ver.
            if self.conv.bias is not None:
                nn.init.constant_(self.conv.bias, 0.)
        elif args.stiefel or args.parainitst:
            nn.init.orthogonal_(self.conv.weight.view(self.conv.weight.shape[0], -1))
            self.conv.weight.data.div_(
                self.conv.weight.data.view(self.conv.weight.shape[0], -1).norm(dim=1).view(-1, 1, 1,
                                                                                           1))  # one-line ver.
            if self.conv.bias is not None:
                nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x):
        x = self.conv(x)
        x *= self.scale.expand(x.shape[0], x.shape[1]).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x


class Conv2dScaledWithBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, args=None):
        super(Conv2dScaledWithBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scale = nn.Parameter(torch.ones(out_channels))

        # para init.
        if args is None or args.oblique or args.parainitob:
            nn.init.normal_(self.conv.weight, mean=0., std=1.)
            self.conv.weight.data.div_(
                self.conv.weight.data.view(self.conv.weight.shape[0], -1).norm(dim=1).view(-1, 1, 1,
                                                                                           1))  # one-line ver.
            if self.conv.bias is not None:
                nn.init.constant_(self.conv.bias, 0.)
        elif args.stiefel or args.parainitst:
            nn.init.orthogonal_(self.conv.weight.view(self.conv.weight.shape[0], -1))
            self.conv.weight.data.div_(
                self.conv.weight.data.view(self.conv.weight.shape[0], -1).norm(dim=1).view(-1, 1, 1,
                                                                                           1))  # one-line ver.
            if self.conv.bias is not None:
                nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x *= self.scale.expand(x.shape[0], x.shape[1]).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x


class LinearScaled(nn.Module):
    def __init__(self, in_features, out_features, bias=True, args=None):
        super(LinearScaled, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias)
        self.scale = nn.Parameter(torch.ones(out_features))
        # para init.
        if args is None or args.oblique:
            nn.init.normal_(self.fc.weight, mean=0., std=1.)
            self.fc.weight.data.div_(self.fc.weight.data.norm(dim=1).view(-1, 1))
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias, 0.)
        elif args.stiefel:
            nn.init.orthogonal_(self.fc.weight)
            self.fc.weight.data.div_(self.fc.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias, 0.)

    def forward(self, x):
        x = self.fc(x)
        x *= self.scale
        return x

# currently not used
# from torch.distributions import Normal
#
# normal = Normal(0, 1)
#
#
# class Conv2dScaledNormal(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True, args=None):
#         super(Conv2dScaledNormal, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
#         self.scale = nn.Parameter(torch.zeros(out_channels))
#         # para init.
#         if args is None or args.oblique:
#             nn.init.normal_(self.conv.weight, mean=0., std=1.)
#             self.conv.weight.data.div_(
#                 self.conv.weight.data.view(self.conv.weight.shape[0], -1).norm(dim=1).view(-1, 1, 1,
#                                                                                            1))  # one-line ver.
#             if self.conv.bias is not None:
#                 nn.init.constant_(self.conv.bias, 0.)
#         elif args.stiefel:
#             nn.init.orthogonal_(self.conv.weight.view(self.conv.weight.shape[0], -1))
#             self.conv.weight.data.div_(
#                 self.conv.weight.data.view(self.conv.weight.shape[0], -1).norm(dim=1).view(-1, 1, 1,
#                                                                                            1))  # one-line ver.
#             if self.conv.bias is not None:
#                 nn.init.constant_(self.conv.bias, 0.)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x *= 2 * normal.cdf(self.scale).expand(x.shape[0], x.shape[1]).unsqueeze(2).unsqueeze(3).expand_as(x)
#         return x
#
#
# class LinearScaledNormal(nn.Module):
#     def __init__(self, in_features, out_features, bias=True, args=None):
#         super(LinearScaledNormal, self).__init__()
#         self.fc = nn.Linear(in_features, out_features, bias)
#         self.scale = nn.Parameter(torch.zeros(out_features))
#         # para init.
#         if args is None or args.oblique:
#             nn.init.normal_(self.fc.weight, mean=0., std=1.)
#             self.fc.weight.data.div_(self.fc.weight.data.norm(dim=1).view(-1, 1))
#             if self.fc.bias is not None:
#                 nn.init.constant_(self.fc.bias, 0.)
#         elif args.stiefel:
#             nn.init.orthogonal_(self.fc.weight)
#             self.fc.weight.data.div_(self.fc.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.
#             if self.fc.bias is not None:
#                 nn.init.constant_(self.fc.bias, 0.)
#
#     def forward(self, x):
#         x = self.fc(x)
#         x *= 2 * normal.cdf(self.scale)
#         return x

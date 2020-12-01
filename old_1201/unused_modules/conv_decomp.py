import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np

default_threshold_rate = 0.1
epsilon = 1e-6


class Conv2dDecomp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, args=None):
        super(Conv2dDecomp, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # self.scale = nn.Parameter(torch.ones(out_channels))

        if args is None:
            decomp_type = 'norm'
        else:
            decomp_type = args.decomposition
        # para init.
        w = getattr(self.conv, 'weight')
        # g is on Oblique/Stiefel manifold
        g = torch.rand(out_channels, in_channels)
        if args is None or args.oblique:
            nn.init.normal_(g)
            g.div_(g.norm(dim=1).view(-1, 1))
        else:  # Stiefel
            nn.init.orthogonal_(g)
            g.div_(g.norm(dim=1).view(-1, 1))
        self.g = Parameter(g.data)
        v = torch.rand_like(w)
        if decomp_type == 'norm':
            # each k*k in v has norm 1
            nn.init.normal_(v)
            v_norm = v.norm(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand_as(v)
            v.div_(v_norm)
        elif decomp_type == 'ob':
            v_f = v.view(-1, v.shape[2], v.shape[3])
            for f in v_f:  # f has shape (k, k) and is on Oblique manifold
                nn.init.normal_(f)
                f.div_(f.norm(dim=1).view(-1, 1))
        elif decomp_type == 'st':
            # each k*k in v is on Stiefel manifold
            v_f = v.view(-1, v.shape[2], v.shape[3])
            for f in v_f:  # f has shape (k, k) and is on Stiefel manifold
                nn.init.orthogonal_(f)
                f.div_(f.norm(dim=1).view(-1, 1))
        self.v = Parameter(v.data)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.)
        del self.conv._parameters['weight']
        self.decomp_type = decomp_type
        # currently only support square kernel
        self.k = kernel_size

    def _setweights(self):
        w = self.v.mul(self.g.unsqueeze(-1).unsqueeze(-1).expand_as(self.v))
        if self.decomp_type != 'norm':  # Oblique, Stiefel, SO3
            w.div_(np.sqrt(self.k))
        setattr(self.conv, 'weight', w)

    def forward(self, x):
        self._setweights()
        x = self.conv(x)
        return x


class Conv2dDecompScaled(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, args=None):
        super(Conv2dDecompScaled, self).__init__()
        self.conv = Conv2dDecomp(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, args=args)
        self.scale = nn.Parameter(torch.ones(out_channels))

    def forward(self, x):
        x = self.conv(x)
        x *= self.scale.expand(x.shape[0], x.shape[1]).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x


class Conv2dDecompScaledWithBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, args=None):
        super(Conv2dDecompScaledWithBN, self).__init__()
        self.conv = Conv2dDecomp(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, args=args)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scale = nn.Parameter(torch.ones(out_channels))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x *= self.scale.expand(x.shape[0], x.shape[1]).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x


class Conv2dDecompScaledPruned(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, args=None):
        super(Conv2dDecompScaledPruned, self).__init__()
        self.conv = Conv2dDecomp(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, args=args)
        self.scale = nn.Parameter(torch.ones(out_channels))
        # self.last_mask = torch.ones(out_channels)
        if args is None:
            self.threshold_rate = default_threshold_rate
        else:
            self.threshold_rate = args.thresholdrate

    def forward(self, x):
        x = self.conv(x)
        active = (self.scale.data > epsilon).float()
        scale_mean_active = self.scale.data.mul_(active).sum().div(active.sum())
        # scale_mean = self.scale.data.mul(self.last_mask).sum().div(self.last_mask.sum())
        scale_mask = (self.scale.data > scale_mean_active * self.threshold_rate).float()
        # self.last_mask = scale_mask.data
        scale_after_pruning = self.scale.mul(scale_mask)
        x *= scale_after_pruning.expand(x.shape[0], x.shape[1]).unsqueeze(2).unsqueeze(3).expand_as(x)
        self.scale.data.mul_(scale_mask)
        return x


class Conv2dDecompScaledPrunedWithBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, args=None):
        super(Conv2dDecompScaledPrunedWithBN, self).__init__()
        self.conv = Conv2dDecomp(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, args=args)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scale = nn.Parameter(torch.ones(out_channels))
        # self.last_mask = torch.ones(out_channels)
        if args is None:
            self.threshold_rate = default_threshold_rate
        else:
            self.threshold_rate = args.thresholdrate

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        active = (self.scale.data > epsilon).float()
        scale_mean_active = self.scale.data.mul_(active).sum().div(active.sum())
        # scale_mean = self.scale.data.mul(self.last_mask).sum().div(self.last_mask.sum())
        scale_mask = (self.scale.data > scale_mean_active * self.threshold_rate).float()
        # self.last_mask = scale_mask.data
        scale_after_pruning = self.scale.mul(scale_mask)
        x *= scale_after_pruning.expand(x.shape[0], x.shape[1]).unsqueeze(2).unsqueeze(3).expand_as(x)
        self.scale.data.mul_(scale_mask)
        return x


class Conv2dWN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, args=None):
        super(Conv2dWN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # self.scale = nn.Parameter(torch.ones(out_channels))

        # para init.
        w = getattr(self.conv, 'weight')
        g = torch.rand(out_channels, in_channels)
        if args is None or args.oblique or args.parainitob:
            nn.init.normal_(g)
            g.div_(g.norm(dim=1).view(-1, 1))
        elif args.stiefel or args.parainitst:
            nn.init.orthogonal_(g)
            g.div_(g.norm(dim=1).view(-1, 1))
        else:  # normal SGD
            nn.init.normal_(g, 0., 0.1)
        self.g = Parameter(g.data)
        v = torch.rand_like(w)
        nn.init.normal_(v)
        v_norm = v.norm(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand_as(v)
        v.div_(v_norm)
        self.v = Parameter(v.data)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.)
        del self.conv._parameters['weight']
        # currently only support square kernel
        self.k = kernel_size

    def _setweights(self):
        v_norm = self.v.norm(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand_as(self.v)
        w = self.v.div(v_norm).mul(self.g.unsqueeze(-1).unsqueeze(-1).expand_as(self.v))
        setattr(self.conv, 'weight', w)

    def forward(self, x):
        self._setweights()
        x = self.conv(x)
        return x


class Conv2dWNScaled(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, args=None):
        super(Conv2dWNScaled, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.scale = nn.Parameter(torch.ones(out_channels))

        # para init.
        w = getattr(self.conv, 'weight')
        g = torch.rand(out_channels, in_channels)
        if args is None or args.oblique or args.parainitob:
            nn.init.normal_(g)
            g.div_(g.norm(dim=1).view(-1, 1))
        elif args.stiefel or args.parainitst:
            nn.init.orthogonal_(g)
            g.div_(g.norm(dim=1).view(-1, 1))
        else:  # normal SGD
            nn.init.normal_(g, 0., 0.1)
        self.g = Parameter(g.data)
        v = torch.rand_like(w)
        nn.init.normal_(v)
        v_norm = v.norm(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand_as(v)
        v.div_(v_norm)
        self.v = Parameter(v.data)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.)
        del self.conv._parameters['weight']
        # currently only support square kernel
        self.k = kernel_size

    def _setweights(self):
        v_norm = self.v.norm(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand_as(self.v)
        w = self.v.div(v_norm).mul(self.g.unsqueeze(-1).unsqueeze(-1).expand_as(self.v))
        setattr(self.conv, 'weight', w)

    def forward(self, x):
        self._setweights()
        x = self.conv(x)
        x *= self.scale.expand(x.shape[0], x.shape[1]).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x


class Conv2dWNScaledWithBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, args=None):
        super(Conv2dWNScaledWithBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scale = nn.Parameter(torch.ones(out_channels))

        # para init.
        w = getattr(self.conv, 'weight')
        g = torch.rand(out_channels, in_channels)
        if args is None or args.oblique or args.parainitob:
            nn.init.normal_(g)
            g.div_(g.norm(dim=1).view(-1, 1))
        elif args.stiefel or args.parainitst:
            nn.init.orthogonal_(g)
            g.div_(g.norm(dim=1).view(-1, 1))
        else:  # normal SGD
            nn.init.normal_(g, 0., 0.1)
        self.g = Parameter(g.data)
        v = torch.rand_like(w)
        nn.init.normal_(v)
        v_norm = v.norm(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand_as(v)
        v.div_(v_norm)
        self.v = Parameter(v.data)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.)
        del self.conv._parameters['weight']
        # currently only support square kernel
        self.k = kernel_size

    def _setweights(self):
        v_norm = self.v.norm(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand_as(self.v)
        w = self.v.div(v_norm).mul(self.g.unsqueeze(-1).unsqueeze(-1).expand_as(self.v))
        setattr(self.conv, 'weight', w)

    def forward(self, x):
        self._setweights()
        x = self.conv(x)
        x = self.bn(x)
        x *= self.scale.expand(x.shape[0], x.shape[1]).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x


class Conv2dWNScaledPruned(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, args=None):
        super(Conv2dWNScaledPruned, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.scale = nn.Parameter(torch.ones(out_channels))
        if args is None:
            self.threshold_rate = default_threshold_rate
        else:
            self.threshold_rate = args.thresholdrate
        # para init.
        w = getattr(self.conv, 'weight')
        g = torch.rand(out_channels, in_channels)
        if args is None or args.oblique or args.parainitob:
            nn.init.normal_(g)
            g.div_(g.norm(dim=1).view(-1, 1))
        elif args.stiefel or args.parainitst:
            nn.init.orthogonal_(g)
            g.div_(g.norm(dim=1).view(-1, 1))
        else:  # normal SGD
            nn.init.normal_(g, 0., 0.1)
        self.g = Parameter(g.data)
        v = torch.rand_like(w)
        nn.init.normal_(v)
        v_norm = v.norm(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand_as(v)
        v.div_(v_norm)
        self.v = Parameter(v.data)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.)
        del self.conv._parameters['weight']
        # currently only support square kernel
        self.k = kernel_size

    def _setweights(self):
        v_norm = self.v.norm(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand_as(self.v)
        w = self.v.div(v_norm).mul(self.g.unsqueeze(-1).unsqueeze(-1).expand_as(self.v))
        setattr(self.conv, 'weight', w)

    def forward(self, x):
        self._setweights()
        x = self.conv(x)
        active = (self.scale.data > epsilon).float()
        scale_mean_active = self.scale.data.mul_(active).sum().div(active.sum())
        scale_mask = (self.scale.data > scale_mean_active * self.threshold_rate).float()
        scale_after_pruning = self.scale.mul(scale_mask)
        x *= scale_after_pruning.expand(x.shape[0], x.shape[1]).unsqueeze(2).unsqueeze(3).expand_as(x)
        self.scale.data.mul_(scale_mask)
        return x


class Conv2dWNScaledPrunedWithBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, args=None):
        super(Conv2dWNScaledPrunedWithBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scale = nn.Parameter(torch.ones(out_channels))
        if args is None:
            self.threshold_rate = default_threshold_rate
        else:
            self.threshold_rate = args.thresholdrate
        # para init.
        w = getattr(self.conv, 'weight')
        g = torch.rand(out_channels, in_channels)
        if args is None or args.oblique or args.parainitob:
            nn.init.normal_(g)
            g.div_(g.norm(dim=1).view(-1, 1))
        elif args.stiefel or args.parainitst:
            nn.init.orthogonal_(g)
            g.div_(g.norm(dim=1).view(-1, 1))
        else:  # normal SGD
            nn.init.normal_(g, 0., 0.1)
        self.g = Parameter(g.data)
        v = torch.rand_like(w)
        nn.init.normal_(v)
        v_norm = v.norm(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand_as(v)
        v.div_(v_norm)
        self.v = Parameter(v.data)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.)
        del self.conv._parameters['weight']
        # currently only support square kernel
        self.k = kernel_size

    def _setweights(self):
        v_norm = self.v.norm(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand_as(self.v)
        w = self.v.div(v_norm).mul(self.g.unsqueeze(-1).unsqueeze(-1).expand_as(self.v))
        setattr(self.conv, 'weight', w)

    def forward(self, x):
        self._setweights()
        x = self.conv(x)
        x = self.bn(x)
        active = (self.scale.data > epsilon).float()
        scale_mean_active = self.scale.data.mul_(active).sum().div(active.sum())
        scale_mask = (self.scale.data > scale_mean_active * self.threshold_rate).float()
        scale_after_pruning = self.scale.mul(scale_mask)
        x *= scale_after_pruning.expand(x.shape[0], x.shape[1]).unsqueeze(2).unsqueeze(3).expand_as(x)
        self.scale.data.mul_(scale_mask)
        return x

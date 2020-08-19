""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
# from dataset import *
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import sys
from conf import settings
from utils import get_network, get_test_dataloader
from pruned_model_para_counting import para_counting
from scaled_for_pruning import default_threshold_rate

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--net', type=str, required=True, help='net type')
    parser.add_argument('-w', '--weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-g', '--gpu', type=bool, default=torch.cuda.is_available(), help='use gpu or not')
    parser.add_argument('-nw', '--numworkers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', '--batchsize', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-shf', '--shuffle', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-t', '--thresholdrate', type=float, default=default_threshold_rate,
                        help='threshold rate used for pruning')
    mani_group = parser.add_mutually_exclusive_group()
    mani_group.add_argument('-o', '--oblique', action='store_true', help='use optimizer on oblique manifold')
    mani_group.add_argument('-s', '--stiefel', action='store_true', help='use optimizer on stiefel manifold')
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument('-p', '--parainitob', action='store_true', help='weight initialization on oblique manifold')
    init_group.add_argument('-P', '--parainitst', action='store_true', help='weight initialization on stiefel manifold')
    parser.add_argument('-f', '--feedback', action='store_true', help='use feedback integrator')
    args = parser.parse_args()

    net = get_network(args, args.gpu)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.numworkers,
        batch_size=args.batchsize,
        shuffle=args.shuffle
    )
    if not args.gpu:
        net.load_state_dict(torch.load(args.weights, map_location='cpu'), args.gpu)
    else:
        net.load_state_dict(torch.load(args.weights), args.gpu)
    # print(net)

    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    for n_iter, (image, label) in enumerate(cifar100_test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
        if args.gpu:
            image = Variable(image).cuda()
            label = Variable(label).cuda()
        output = net(image)
        _, pred = output.topk(5, 1, largest=True, sorted=True)

        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        # compute top 5
        correct_5 += correct[:, :5].sum()

        # compute top1
        correct_1 += correct[:, :1].sum()

    print()
    print("Top 1 err: ", (1 - correct_1 / len(cifar100_test_loader.dataset)).tolist())
    print("Top 5 err: ", (1 - correct_5 / len(cifar100_test_loader.dataset)).tolist())
    # print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
    # full_para_num = sum(p.numel() for p in net.parameters())
    # pruned_para_num = para_counting(net, args)
    # pruning_rate = pruned_para_num / full_para_num
    # print("Full parameter numbers:", full_para_num)
    # print("Parameter numbers:", pruned_para_num)
    # print("Pruning rate:", pruning_rate)
    # print("Parameter numbers: {}".format(para_counting(net)))

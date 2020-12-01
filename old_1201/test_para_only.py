""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
import torch
from utils import get_network
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
    if not args.gpu:
        net.load_state_dict(torch.load(args.weights, map_location='cpu'), args.gpu)
    else:
        net.load_state_dict(torch.load(args.weights), args.gpu)
    # print(net)

    # full_para_num = sum(p.numel() for p in net.parameters())
    pruned_para_num = para_counting(net, args)
    # pruning_rate = pruned_para_num / full_para_num
    # print("Full parameter numbers:", full_para_num)
    print("Parameter numbers:", pruned_para_num)
    # print("Pruning rate:", pruning_rate)

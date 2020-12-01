""" configurations for this project
author baiyu
"""
from datetime import datetime

# mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2470, 0.2435, 0.2616)

SVHN_TRAIN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_TRAIN_STD = (0.1980, 0.2010, 0.1970)

IMAGENET_TRAIN_MEAN = (0.485, 0.456, 0.406)
IMAGENET_TRAIN_STD = (0.229, 0.224, 0.225)

# $ python get_mean_std.py --dataset CIFAR10
# mean: tensor([0.4914, 0.4822, 0.4465])
# std: tensor([0.2470, 0.2435, 0.2616])
# $ python get_mean_std.py --dataset CIFAR100
# mean: tensor([0.5071, 0.4865, 0.4409])
# std: tensor([0.2673, 0.2564, 0.2762])
# $ python get_mean_std.py --dataset SVHN
# mean: tensor([0.4377, 0.4438, 0.4728])
# std: tensor([0.1980, 0.2010, 0.1970])
# $ python get_mean_std.py --dataset FashionMNIST
# mean: tensor([0.2860])
# std: tensor([0.3530])
# $ python get_mean_std.py --dataset MNIST
# mean: tensor([0.1307])
# std: tensor([0.3081])

# directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

# tensorboard log dir
LOG_DIR = 'runs'

# total training epochs
# EPOCH = 200
# MILESTONES = [60, 120, 160]
# EPOCH = 80
# MILESTONES = [10, 20, 30, 40, 50, 60, 70]
# EPOCH = 100
# MILESTONES = [20, 40, 60, 80, 90]
EPOCH = 160
MILESTONES = [20, 40, 60, 80, 100, 120, 140]

# time of we run the script
TIME_NOW = ''.join(list(filter(str.isdigit, datetime.now().isoformat()[:-6])))

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10

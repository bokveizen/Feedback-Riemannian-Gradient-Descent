import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sgd_original import SGD
from sgd_oblique import SGDOblique
from sgd_stiefel import SGDStiefel
import numpy as np
import argparse
import time
from datetime import datetime
import sys
from tensorboardX import SummaryWriter
import os

CUR_TIME = ''.join(list(filter(str.isdigit, datetime.now().isoformat()[:-6])))
CMD = ' '.join(sys.argv[1:])

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--network', type=str, default='simple', help="the network to use",
                    choices=['simple', 'relu', 'bn', 'relus'])
parser.add_argument('-e', '--epochs', type=int, default=10, help="the number of epochs")
parser.add_argument('-b', '--batchsize', type=int, default=64, help="batch size")
parser.add_argument('-l', '--learningrate', type=float, default=0.02, help="learning rate")
parser.add_argument('-lf', '--lossprintf', type=int, default=1, help="print the loss every [lossprintf] batches")
parser.add_argument('-ef', '--evaluationf', type=int, default=1, help="do evaluation every [evaluationf] epochs")
parser.add_argument('-nw', '--numworkers', type=int, default=0, help="the num of workers for dataloader")
parser.add_argument('-ld', '--logdir', type=str, default='log_mlp_{}.txt'.format(CMD), help="the path of log file")
mani_group = parser.add_mutually_exclusive_group()
mani_group.add_argument('-o', '--oblique', action='store_true', help='use optimizer on oblique manifold')
mani_group.add_argument('-s', '--stiefel', action='store_true', help='use optimizer on stiefel manifold')
init_group = parser.add_mutually_exclusive_group()
init_group.add_argument('-p', '--parainitob', action='store_true', help='weight initialization on oblique manifold')
init_group.add_argument('-P', '--parainitst', action='store_true', help='weight initialization on stiefel manifold')
parser.add_argument('-f', '--feedback', action='store_true', help="use feedback integrator")
args = parser.parse_args()

f = open(args.logdir, 'a+')


class ModelSimple(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(ModelSimple, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        if args.oblique or args.parainitob:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0., std=1.)
                    m.weight.data.div_(m.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.
        elif args.stiefel or args.parainitst:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    m.weight.data.div_(m.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class ModelReLU(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(ModelReLU, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        if args.oblique or args.parainitob:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0., std=1.)
                    m.weight.data.div_(m.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.
        elif args.stiefel or args.parainitst:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    m.weight.data.div_(m.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.

    def forward(self, x):
        x = self.layer1(x).relu()
        x = self.layer2(x).relu()
        x = self.layer3(x)
        return x


class ModelReLUScaled(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(ModelReLUScaled, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.scale1 = nn.Parameter(torch.ones(n_hidden_1))
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.scale2 = nn.Parameter(torch.ones(n_hidden_2))
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        self.scale3 = nn.Parameter(torch.ones(out_dim))
        if args.oblique or args.parainitob:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0., std=1.)
                    m.weight.data.div_(m.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.
        elif args.stiefel or args.parainitst:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    m.weight.data.div_(m.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.

    def forward(self, x):
        x = self.layer1(x)
        x *= self.scale1
        x = x.relu()
        x = self.layer2(x)
        x *= self.scale2
        x = x.relu()
        x = self.layer3(x)
        x *= self.scale3
        return x


class ModelBN(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(ModelBN, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.bn1 = nn.BatchNorm1d(n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.bn2 = nn.BatchNorm1d(n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        if args.oblique or args.parainitob:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0., std=1.)
                    m.weight.data.div_(m.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.
        elif args.stiefel or args.parainitst:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    m.weight.data.div_(m.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x).relu()
        x = self.layer2(x)
        x = self.bn2(x).relu()
        x = self.layer3(x)
        return x


def dist_from_ob(p):
    return (p.data.mm(p.data.t()) - torch.eye(p.data.shape[0], device=p.device)).diag().norm()


def dist_from_st(p):
    return (p.data.mm(p.data.t()) - torch.eye(p.data.shape[0], device=p.device)).norm()

batch_size = args.batchsize
learning_rate = args.learningrate
epochs = args.epochs

# data_tf = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize(mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
#                           std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628])])

# data_tf = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
#                           std=[0.1980, 0.2010, 0.1970])])

data_tf = transforms.Compose([transforms.ToTensor()])

# data_tf = transforms.Compose(
#     [transforms.ToTensor(),
#      # transforms.Normalize(mean=0.1307, std=0.3081)
#      ])

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

train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf, download=True)

# train_dataset = datasets.CIFAR10(root='./data', train=True, transform=data_tf, download=True)
# test_dataset = datasets.CIFAR10(root='./data', train=False, transform=data_tf, download=True)

# train_dataset = datasets.SVHN(root='./data', split='train', transform=data_tf, download=True)
# test_dataset = datasets.SVHN(root='./data', split='test', transform=data_tf, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.numworkers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.numworkers)

# input_dim = 3 * 32 * 32
input_dim = 28 * 28
hidden_dim = (300, 100)
output_dim = 10
if args.network == 'simple':
    model = ModelSimple(input_dim, hidden_dim[0], hidden_dim[1], output_dim)
elif args.network == 'relu':
    model = ModelReLU(input_dim, hidden_dim[0], hidden_dim[1], output_dim)
elif args.network == 'bn':
    model = ModelBN(input_dim, hidden_dim[0], hidden_dim[1], output_dim)
elif args.network == 'relus':
    model = ModelReLUScaled(input_dim, hidden_dim[0], hidden_dim[1], output_dim)
else:
    model = None
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
if args.oblique:
    optimizer = SGDOblique(model.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4, feedback=args.feedback)
elif args.stiefel:
    optimizer = SGDStiefel(model.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4, feedback=args.feedback)
else:
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# use tensorboard
LOG_DIR = 'mlp_log'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
writer = SummaryWriter(log_dir=os.path.join(
    LOG_DIR, args.logdir))
# input_tensor = torch.Tensor(12, 3, 32, 32)
# writer.add_graph(model, Variable(input_tensor, requires_grad=True))

T_s = time.process_time()
ii = 0
for e in range(epochs):
    i = 0
    for data in train_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
        ii += 1
        w1 = model.layer1.weight.data
        w2 = model.layer2.weight.data
        w3 = model.layer3.weight.data
        writer.add_scalar('Layer1/dist_from_ob', dist_from_ob(w1), ii)
        writer.add_scalar('Layer2/dist_from_ob', dist_from_ob(w2), ii)
        writer.add_scalar('Layer3/dist_from_ob', dist_from_ob(w3), ii)
        writer.add_scalar('Layer1/dist_from_st', dist_from_st(w1), ii)
        writer.add_scalar('Layer2/dist_from_st', dist_from_st(w2), ii)
        writer.add_scalar('Layer3/dist_from_st', dist_from_st(w3), ii)
        writer.add_scalar('Train/loss', loss.item(), ii)
        if args.lossprintf > 0 and i % args.lossprintf == 0:
            print('{}, {}, loss: {:.4}'.format(e, i, loss.data.item()), file=f)
            print('{}, {}, loss: {:.4}'.format(e, i, loss.data.item()))
            w1n = [float(i.norm()) for i in w1]
            w2n = [float(i.norm()) for i in w2]
            w3n = [float(i.norm()) for i in w3]
            print('Layer1 weights norm: mean = {}, var = {}'.format(np.mean(w1n), np.var(w1n)), file=f)
            print('Layer2 weights norm: mean = {}, var = {}'.format(np.mean(w2n), np.var(w2n)), file=f)
            print('Layer3 weights norm: mean = {}, var = {}'.format(np.mean(w3n), np.var(w3n)), file=f)
            print('Layer1 weights norm: mean = {}, var = {}'.format(np.mean(w1n), np.var(w1n)))
            print('Layer2 weights norm: mean = {}, var = {}'.format(np.mean(w2n), np.var(w2n)))
            print('Layer3 weights norm: mean = {}, var = {}'.format(np.mean(w3n), np.var(w3n)))
    if (args.evaluationf > 0 and ((e + 1) % args.evaluationf == 0)) or e == epochs - 1:
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for data in test_loader:
            img, label = data
            img = img.view(img.size(0), -1)
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.data.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()
        print('Evaluation @ epoch #{}, Test Loss: {:.6f}, Acc: {:.6f}'.format(
            e,
            eval_loss / (len(test_dataset)),
            eval_acc / (len(test_dataset))
        ), file=f)
        print('Evaluation @ epoch #{}, Test Loss: {:.6f}, Acc: {:.6f}'.format(
            e,
            eval_loss / (len(test_dataset)),
            eval_acc / (len(test_dataset))))
T_e = time.process_time()
print('Running Time:{} ms'.format((T_e - T_s) * 1000), file=f)
print('Running Time:{} ms'.format((T_e - T_s) * 1000))
# w1 = model.layer1.weight.data
# w1n = [float(i.norm()) for i in w1]
# w2 = model.layer2.weight.data
# w2n = [float(i.norm()) for i in w2]
# w3 = model.layer3.weight.data
# w3n = [float(i.norm()) for i in w3]
#
# print('Layer1 weights norm: mean = {}, var = {}'.format(np.mean(w1n), np.var(w1n)), file=f)
# print('Layer2 weights norm: mean = {}, var = {}'.format(np.mean(w2n), np.var(w2n)), file=f)
# print('Layer3 weights norm: mean = {}, var = {}'.format(np.mean(w3n), np.var(w3n)), file=f)

f.close()
writer.close()
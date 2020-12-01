import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sgd_original import SGD
# from sgd_oblique import SGDOblique
# from sgd_stiefel import SGDStiefel
# from sgd_stiefel_decomp import SGDStiefelDecomp
# from sgd_oblique_decomp import SGDObliqueDecomp
from sgd_oblique_wn2 import SGDObliqueWN
from sgd_stiefel_wn2 import SGDStiefelWN
from sgd_oblique_kk import SGDObliqueKK
from sgd_stiefel_kk import SGDStiefelKK
from conv_decomp import Conv2dWN
import numpy as np
import argparse
import time
from datetime import datetime
import sys

CUR_TIME = ''.join(list(filter(str.isdigit, datetime.now().isoformat()[:-6])))
CMD = ' '.join(sys.argv[1:])

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--network', type=str, default='simple', help="the network to use",
                    choices=['simple', 'scaled'])
parser.add_argument('-e', '--epochs', type=int, default=10, help="the number of epochs")
parser.add_argument('-b', '--batchsize', type=int, default=64, help="batch size")
parser.add_argument('-l', '--learningrate', type=float, default=0.02, help="learning rate")
parser.add_argument('-lf', '--lossprintf', type=int, default=0, help="print the loss every [lossprintf] batches")
parser.add_argument('-ef', '--evaluationf', type=int, default=1, help="do evaluation every [evaluationf] epochs")
parser.add_argument('-nw', '--numworkers', type=int, default=0, help="the num of workers for dataloader")
parser.add_argument('-ld', '--logdir', type=str, default='log_fwu_0804{}.txt'.format(CMD), help="the path of log file")
parser.add_argument('-stp', '--stiefelpunishment', type=float, default=0.,
                    help='Punishment factor used for stiefel manifold')
parser.add_argument('-d', '--decomposition', type=str, default='',
                    help='use decomposition for conv weights on stiefel manifold')
parser.add_argument('-o', '--oblique', action='store_true', help="use optimizer on oblique manifold")
parser.add_argument('-s', '--stiefel', action='store_true', help="use optimizer on stiefel manifold")
parser.add_argument('-p', '--parainit', action='store_true', help="do parameter initialization on oblique manifold")
parser.add_argument('-f', '--feedback', action='store_true', help="use feedback integrator")
args = parser.parse_args()

f = open(args.logdir, 'a+')


class ModelCNNSimple(nn.Module):
    def __init__(self, para_init=args.parainit):
        super(ModelCNNSimple, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU(True)
        if para_init:
            self.para_init()

    def para_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, mean=0., std=1.)
                # m.weight.data.div_(m.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.
                nn.init.orthogonal_(m.weight)
                m.weight.data.div_(m.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, mean=0., std=1.)
                nn.init.orthogonal_(m.weight.view(m.weight.shape[0], -1))
                m.weight.data.div_(
                    m.weight.data.view(m.weight.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1))  # one-line ver.
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):  # [4,3,32,32]
        x = self.pool(self.relu(self.conv1(x)))  # [4,6,14,14]
        x = self.pool(self.relu(self.conv2(x)))  # [4,16,5,5]
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModelCNNScaled(nn.Module):
    def __init__(self, para_init=args.parainit):
        super(ModelCNNScaled, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1scale = nn.Parameter(torch.ones(6))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2scale = nn.Parameter(torch.ones(16))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1scale = nn.Parameter(torch.ones(120))
        self.fc2 = nn.Linear(120, 84)
        self.fc2scale = nn.Parameter(torch.ones(84))
        self.fc3 = nn.Linear(84, 10)
        self.fc3scale = nn.Parameter(torch.ones(10))
        self.relu = nn.ReLU(True)
        if para_init:
            self.para_init()

    def para_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=1.)
                m.weight.data.div_(m.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.
                # nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0., std=1.)
                m.weight.data.div_(
                    m.weight.data.view(m.weight.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1))  # one-line ver.
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):  # [4,3,32,32]
        # x = self.pool(self.relu(self.conv1(x)))  # [4,6,14,14]
        x = self.conv1(x)
        x *= self.conv1scale.expand(x.shape[0], x.shape[1]).unsqueeze(2).unsqueeze(3).expand_as(x)
        x = self.pool(self.relu(x))
        # x = self.pool(self.relu(self.conv2(x)))  # [4,16,5,5]
        x = self.conv2(x)
        x *= self.conv2scale.expand(x.shape[0], x.shape[1]).unsqueeze(2).unsqueeze(3).expand_as(x)
        x = self.pool(self.relu(x))
        x = x.view(-1, 16 * 5 * 5)
        # x = self.relu(self.fc1(x))
        x = self.fc1(x)
        # x *= self.fc1scale
        x = self.relu(x)
        # x = self.relu(self.fc2(x))
        x = self.fc2(x)
        # x *= self.fc2scale
        x = self.relu(x)
        x = self.fc3(x)
        # x *= self.fc3scale
        return x


class ModelCNN2dWN(nn.Module):
    def __init__(self, para_init=args.parainit):
        super(ModelCNN2dWN, self).__init__()
        self.conv1 = Conv2dWN(3, 6, 5, args=args)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = Conv2dWN(6, 16, 5, args=args)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU(True)
        self.para_init()

    def para_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                m.weight.data.div_(m.weight.data.norm(dim=1).view(-1, 1))  # one-line ver.
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):  # [4,3,32,32]
        x = self.pool(self.relu(self.conv1(x)))  # [4,6,14,14]
        x = self.pool(self.relu(self.conv2(x)))  # [4,16,5,5]
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


batch_size = args.batchsize
learning_rate = args.learningrate
epochs = args.epochs

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                          std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628])])

train_dataset = datasets.CIFAR10(
    root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.numworkers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.numworkers)

model = ModelCNN2dWN()

if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
if args.oblique:
    # optimizer = SGDOblique(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4,
    #                        feedback=args.feedback)
    optimizer = SGDObliqueWN(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4,
                             feedback=args.feedback)
elif args.stiefel:
    # optimizer = SGDStiefel(model.parameters(), lr=learning_rate, momentum=0.9, feedback=args.feedback)
    optimizer = SGDStiefelWN(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4,
                             feedback=args.feedback)
else:
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

T_s = time.process_time()
for e in range(epochs):
    i = 0
    for data in train_loader:
        img, label = data
        # img = img.view(img.size(0), -1)
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
        if args.lossprintf > 0 and i % args.lossprintf == 0:
            print('{}, {}, loss: {:.4}'.format(e, i, loss.data.item()), file=f)
            print('{}, {}, loss: {:.4}'.format(e, i, loss.data.item()))
    scheduler.step()
    if (args.evaluationf > 0 and ((e + 1) % args.evaluationf == 0)) or e == epochs - 1:
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for data in test_loader:
            img, label = data
            # img = img.view(img.size(0), -1)
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
            eval_acc / (len(test_dataset))
        ))

T_e = time.process_time()
print('Running Time:{} ms'.format((T_e - T_s) * 1000), file=f)
print('Running Time:{} ms'.format((T_e - T_s) * 1000))
f.close()

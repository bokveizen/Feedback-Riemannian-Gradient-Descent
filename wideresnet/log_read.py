from collections import defaultdict
import matplotlib.pyplot as plt
import os
import os.path as osp

fig1 = plt.figure()
ax1 = fig1.add_subplot()
fig2 = plt.figure()
ax2 = fig2.add_subplot()
fig3 = plt.figure()
ax3 = fig3.add_subplot()
fig4 = plt.figure()
ax4 = fig4.add_subplot()
path = 'log0928'
for lf in os.listdir(path):
    # f = open('logtxt/log ({}).txt'.format(i))
    f = open(osp.join(path, lf))
    lines = f.readlines()
    dic = defaultdict(list)
    for line in lines:
        info = line[13:-1]
        items = info.split(', ')
        for item in items:
            key, value = item.split(': ')
            key = key[1:-1]
            if value[0] == '"':
                value = value[1:-1]
            dic[key].append(value)
    optim_method = dic['optim_method'][0]
    lr = dic['lr'][0]
    lrg = dic['lrg'][0]
    train_loss = [float(loss) for loss in dic['train_loss']]
    train_acc = [float(acc) for acc in dic['train_acc']]
    test_loss = [float(loss) for loss in dic['test_loss']]
    test_acc = [float(acc) for acc in dic['test_acc']]
    if 'ST' in optim_method or 'SGD' in optim_method:
        label = optim_method + '_' + lr + '_' + lrg
        ax1.plot(train_loss, label=label)
        ax2.plot(train_acc, label=label)
        ax3.plot(test_loss, label=label)
        ax4.plot(test_acc, label=label)
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.show()




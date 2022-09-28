# -*- coding: utf-8 -*-
# @Time : 2022/9/28 10:36
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : write_read_files.py
# @Project : LiMuAI
import torch
from torch import nn
from torch.nn import functional as F


def load_save_param():
    x = torch.arange(4)
    torch.save(x, 'x-file')

    x2 = torch.load('x-file')
    print(x2)

    y = torch.zeros(4)
    torch.save([x, y], 'xy-file')

    X,Y = torch.load('xy-file')
    print(X,Y)

    mydict = {'x': x, 'y': y}
    torch.save(mydict, 'mydict')
    mydict2 = torch.load('mydict')
    print(mydict2)


def load_save_model():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(20, 256)
            self.output = nn.Linear(256, 10)

        def forward(self, x):
            return self.output(F.relu(self.hidden(x)))

    net = MLP()
    print(net)
    X = torch.randn(size=(2, 20))
    Y = net(X)

    torch.save(net.state_dict(), 'mlp.params')

    clone = MLP()
    clone.load_state_dict(torch.load('mlp.params'))
    print(clone.eval())

    Y_clone = clone(X)
    print(Y_clone == Y)


def main():
    print("---加载和保存张量---")
    #load_save_param()
    print("---加载和保存模型---")
    load_save_model()


if __name__ == '__main__':
    main()
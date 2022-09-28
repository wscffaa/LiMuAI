# -*- coding: utf-8 -*-
# @Time : 2022/9/28 10:27
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : customization_layer.py
# @Project : LiMuAI
import torch
import torch.nn.functional as F
from torch import nn


def no_param_layer():
    class CenteredLayer(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, X):
            return X - X.mean()

    layer = CenteredLayer()
    print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    Y = net(torch.rand(4, 8))
    print(Y.mean())


def with_param_layer():

    class MyLinear(nn.Module):
        def __init__(self, in_units, units):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(in_units, units))
            self.bias = nn.Parameter(torch.randn(units, ))

        def forward(self, X):
            linear = torch.matmul(X, self.weight.data) + self.bias.data
            return F.relu(linear)

    linear = MyLinear(5, 3)
    print(linear.weight)

    linear(torch.rand(2, 5))
    net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
    print(net(torch.rand(2, 64)))


def main():
    print("---不带参数的层---")
    no_param_layer()
    print("---带参数的层---")
    with_param_layer()


if __name__ == '__main__':
    main()
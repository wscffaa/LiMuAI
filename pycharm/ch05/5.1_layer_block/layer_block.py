# -*- coding: utf-8 -*-
# @Time : 2022/9/27 16:48
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : layer_block.py
# @Project : LiMuAI
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    def forward(self, x):
        return self.out(F.relu(self.hidden(x)))


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 18)

    def forward(self, X):
        return self.linear(self.net(X))


def block():
    print("---块的基本功能---")
    print("1.将输入数据作为其前向传播函数的参数")
    print("2.通过前向传播函数来生成输出")
    print("3.计算其输出关于输入的梯度，可通过其反向传播函数进行访问")
    print("4.存储和访问前向传播计算所需的参数")
    print("5.根据需要初始化模型参数")


def main():
    block()
    X = torch.rand(2, 20)
    net = MLP()
    net_1 = NestMLP()
    print(net(X).size())
    print(net_1(X).size())


if __name__ == '__main__':
    main()
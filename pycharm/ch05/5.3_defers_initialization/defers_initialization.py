# -*- coding: utf-8 -*-
# @Time : 2022/9/28 10:24
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : defers_initialization.py
# @Project : LiMuAI

import torch
from torch import nn


def main():
    """延后初始化"""
    net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
    print(net[0].weight)  # 尚未初始化
    print(net)

    X = torch.rand(2, 20)
    net(X)
    print(net)


if __name__ == '__main__':
    main()
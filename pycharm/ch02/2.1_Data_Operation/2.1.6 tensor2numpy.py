#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2022/9/15 上午11:14
@Author  : feifan
@FileName: 2.1.6 tensor2numpy.py
@Software: PyCharm
"""

import torch

def tensor2numpy():
    X = torch.randn(3,4)
    print(type(X))

    Y = X.numpy()
    print(type(Y))

    Z = torch.tensor(Y)
    print(type(Z))

    a = torch.tensor([3.5])
    print(a)
    print(type(a))

    print(a.item())
    print(type(a.item()))

    print(float(a))
    print(type(float(a)))

    print(int(a))
    print(type(int(a)))

def main():
    tensor2numpy()

if __name__ == '__main__':
    main()
#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 下午5:56
# @Author  : feifan
# @FileName: 2.1.1_start.py
# @Software: PyCharm

import torch

def data():

    print("-----------------------")
    print("2.1.1 入门")
    print("-----------------------")
    x = torch.arange(12)
    print(x)
    print("张量的形状为：" + str(x.shape))
    print("张量中元素的总数：" + str(x.numel()))

    X = x.reshape(3,4)
    print(X)

    X = X.reshape(-1,6)
    print("-----------------------")
    print("-1能自动调用计算出维度的功能")
    print("-----------------------")
    print(X)

    x_zero = torch.zeros((1,3,4))
    print("-----------------------")
    print("初始化全零矩阵")
    print("-----------------------")
    print(x_zero)

    x_one = torch.ones((1,4,5))
    print("-----------------------")
    print("初始化全一矩阵")
    print("-----------------------")
    print(x_one)

    x_randn = torch.randn(3,4)
    print("-----------------------")
    print("初始化正态分布矩阵")
    print("-----------------------")
    print(x_randn)

    x_special = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    print("-----------------------")
    print("初始化特定列表元素矩阵")
    print("最外层的列表对应于轴0，内层的列表对应于轴1")
    print("-----------------------")
    print(x_special)

class main():
    data()

if __name__ == '__main__':
    main()
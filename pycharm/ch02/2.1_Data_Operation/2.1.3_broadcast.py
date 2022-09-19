#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 下午7:02
# @Author  : feifan
# @FileName: 2.1.3_broadcast.py
# @Software: PyCharm

import torch

def broadcast():
    x = torch.arange(3).reshape((3,1))
    y = torch.arange(2).reshape((1,2))

    add = x + y
    print(x)
    print(y)
    print("矩阵x复制列，矩阵y复制行，然后按照元素相加")
    print("执行广播机制后的结果如下")
    print(add)

class main():
    print("--------------------------------------------")
    print("2.1.3 广播机制")
    print("在不同形状的任意两个张量上调用广播机制来执行按元素操作")
    print("首先通过适当复制元素来扩展相应数组至相同形状")
    print("然后再对生成的数组执行按元素操作")
    print("--------------------------------------------")
    broadcast()

if __name__ == '__main__':
    main()
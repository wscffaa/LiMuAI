#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 下午8:23
# @Author  : feifan
# @FileName: 2.1.4_index_slice.py
# @Software: PyCharm

import torch

def index():
    print("--------------------------------------------")
    print("2.1.4 索引和切片")
    print("第一个元素的索引是0，最后一个元素索引是-1")
    print("--------------------------------------------")

    a = torch.arange(8)
    print(a)
    print("通常使用【-1】指定最后一个元素")
    print(a[-1])
    print("使用【1:3】指定第二个和第三个元素")
    print(a[1:3])

def slice():
    b = torch.randn(3,4)
    print(b)
    print("第三行第四列元素如下")
    print(b[2,3])
    print("通过[0:2]指定第一二行，[1:3]指定第二三列")
    print(b[0:2,1:3])
    print("通过[：]指定所有列")
    print(b[1:4,:])

def zero_like():
    print("--------------------------------------------")
    print("2.1.5 节省内存")
    print("--------------------------------------------")
    x = torch.randn(2,3)
    y = torch.randn(2,3)

    before = id(x)
    print('id(x_before): ', before)
    x = x + y
    after = id(x)
    print('id(x_after): ', after)
    print(before == after)

    print("--------------------------------------------")
    print("使用切片表示法将操作的结果分配给先前分配的数组")
    print("使用zeros_like来创建全0且与x形状相同的矩阵")
    print("--------------------------------------------")
    Z = torch.zeros_like(x)
    print('id(Z): ', id(Z))
    Z[:] = x + y
    print('id(Z): ', id(Z))


class main():

    index()
    slice()
    zero_like()

if __name__ == '__main__':
    main()
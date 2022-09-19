# -*- coding: utf-8 -*-
# @Time : 2022/9/15 17:12
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : linear_algebra.py
# @Project : LiMuAI


import torch

def scalar():
    print("仅包含一个数值的叫标量（scalar）")
    print("标量由只有一个元素的张量表示")
    x = torch.tensor(3.0)
    y = torch.tensor(2.0)
    print(x + y, x * y, x /y, x ** y)

def element():
    print("一维张量处理向量,张量可以具有任意长度")
    x = torch.arange(4)
    print(x)

    print("列向量是向量的默认方向")
    y = torch.randn(4,1)
    print(y)

def matrices():
    a = torch.arange(20).reshape(4, 5)
    print(a)

    print("矩阵A的转置如下")
    a_t = a.T
    print(a_t)

    print("对称矩阵如下")
    b = torch.tensor([[1,2,3], [2,0,4], [3,4,5]])
    print(b)
    print(b.T)
    print(b == b.T)

def main():
    print("-----本章主要内容是线性代数-----")
    print("0-d（标量）通常代表一个类别")
    print("1-d（向量）通常代表一个特征向量")
    print("2-d（矩阵）通常代表一个样本，即特征矩阵")
    print("3-d通常代表RGB图片，即CHW")
    print("4-d通常代表一个RGB图片批量，即BCHW")
    print("5-d通常代表一个视频批次，即BTCHW")
    scalar()
    element()
    matrices()


if __name__ == '__main__':
    main()
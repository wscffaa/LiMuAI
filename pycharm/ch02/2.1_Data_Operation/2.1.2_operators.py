#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 下午5:57
# @Author  : feifan
# @FileName: 2.1.2_operators.py
# @Software: PyCharm

import torch

def operators():

    print("-----------------------")
    print("2.1.2 运算符")
    print("在同一形状的任意两个张量上调用按元素操作")
    print("-----------------------")

    x = torch.tensor([1.0,2,4,8])
    y = torch.tensor([2,2,2,2])
    print(x)
    print(y)

    add = x + y
    print("-----------------------")
    print("按元素相加")
    print("-----------------------")
    print(add)

    subtract = x - y
    print("-----------------------")
    print("按元素相减")
    print("-----------------------")
    print(subtract)

    multiply = x * y
    print("-----------------------")
    print("按元素相乘")
    print("-----------------------")
    print(multiply)

    division = x / y
    print("-----------------------")
    print("按元素相除")
    print("-----------------------")
    print(division)

    power = x ** y
    print("-----------------------")
    print("按元素求幂")
    print("-----------------------")
    print(power)

    exp = torch.exp(x)
    print("-----------------------")
    print("按元素求指数ln(x)")
    print("-----------------------")
    print(exp)
    print("-----------------------")

def concatenate():
    x = torch.arange(12,dtype=torch.float32).reshape((3,4))
    y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,5,6,7]])
    print(x)
    print(y)

    concat_x = torch.concat((x,y), dim=0)
    print("-----------------------")
    print("沿行连接两个矩阵")
    print("-----------------------")
    print(concat_x)

    concat_y = torch.concat((x,y), dim=1)
    print("-----------------------")
    print("沿列连接两个矩阵")
    print("-----------------------")
    print(concat_y)

    print("-----------------------")
    print("通过逻辑运算符构建二元张量")
    print("-----------------------")
    print(x == y)

    print("-----------------------")
    print("张量所有元素求和")
    print("-----------------------")
    print(x.sum())

class main():
    #operators()
    concatenate()

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
# @Time : 2022/9/27 15:00
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : MLP_basicConcept.py
# @Project : LiMuAI
import torch
from d2l import torch as d2l


def ReLU():
    """ReLU=Max(x, 0)"""
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.relu(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
    d2l.plt.show()

    """ReLU的导数为分段函数，输入为负数时导数为0，输入为正数时导数为1"""
    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
    d2l.plt.show()


def sigmoid():
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.sigmoid(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
    d2l.plt.show()

    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
    d2l.plt.show()


def tanh():
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.tanh(x)

    d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
    d2l.plt.show()

    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
    d2l.plt.show()


def main():
    # ReLU()
    # sigmoid()
    tanh()


if __name__ == '__main__':
    main()

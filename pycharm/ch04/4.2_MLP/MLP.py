# -*- coding: utf-8 -*-
# @Time : 2022/9/27 15:19
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : MLP.py
# @Project : LiMuAI
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)

b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)

b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

loss = nn.CrossEntropyLoss(reduction='none')


num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)


def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)


def net(x):
    x = x.reshape((-1, num_inputs))
    H = relu(x@W1 + b1)
    return H @ W2 + b2


def main():
    print("---多层感知机的从零开始实现---")
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    d2l.predict_ch3(net, test_iter)
    d2l.plt.show()


if __name__ == '__main__':
    main()
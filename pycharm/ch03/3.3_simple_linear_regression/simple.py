# -*- coding: utf-8 -*-
# @Time : 2022/9/26 17:26
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : simple.py
# @Project : LiMuAI
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个Pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def main():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    print("------线性回归的简洁实现------")

    batch_size = 10
    data_iter=load_array((features, labels), batch_size)

    #print(next(iter(data_iter)))

    net = nn.Sequential(nn.Linear(2,1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()

    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3

    """
    1.通过调用net(X)生成预测并计算损失l（前向传播）。

    2.通过进行反向传播来计算梯度。
    
    3.通过调用优化器来更新模型参数。
    """
    for epoch in range(num_epochs):
        for X,y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = net[0].weight.data
    b = net[0].bias.data
    print("w的估计误差： ", true_w - w.reshape(true_w.shape))
    print("b的估计误差： ", true_b - b)

if __name__ == '__main__':
    main()
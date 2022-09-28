# -*- coding: utf-8 -*-
# @Time : 2022/9/28 10:03
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : parameter.py
# @Project : LiMuAI
import torch
from torch import nn


def para1():
    net = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1))

    X = torch.rand(size=(2, 4))

    print(net(X))
    print(net[2].state_dict())
    print(type(net[2].bias))
    print(net[2].bias)
    print(net[2].bias.data)

    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print(*[(name, param.shape) for name, param in net.named_parameters()])


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net


def para2():
    X = torch.rand(size=(2, 4))
    rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
    print(rgnet(X).size())
    print(rgnet)
    print(rgnet[0][1][0].bias)  # 访问第一个块中，第二个子块的第一层的偏置项
    print(rgnet[0][1][2].weight)


def parameter_initialization():

    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.zeros_(m.bias)

    def init_constant(m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.weight, 1)
            nn.init.zeros_(m.bias)

    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    # net.apply(init_normal)
    # net.apply(init_constant)

    print(net[0].weight.data[0], net[0].bias.data[0])

    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    def init_42(m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.weight, 42)

    net[0].apply(init_xavier)
    net[2].apply(init_42)
    print(net[0].weight.data[0])
    print(net[2].weight.data)


def parameter_customization():
    def my_init(m):
        if type(m) == nn.Linear:
            print("Init", *[(name, param.shape)
                            for name, param in m.named_parameters()][0])
            nn.init.uniform_(m.weight, -10, 10)
            m.weight.data *= m.weight.data.abs() >= 5

    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    net.apply(my_init)
    print(net[0].weight[:2])


def parameter_share():
    # 我们需要给共享层一个名称，以便可以引用它的参数
    shared = nn.Linear(8, 8)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                        shared, nn.ReLU(),
                        shared, nn.ReLU(),
                        nn.Linear(8, 1))

    X = torch.rand(size=(2, 4))
    net(X)
    # 检查参数是否相同
    print(net[2].weight.data[0] == net[4].weight.data[0])
    net[2].weight.data[0, 0] = 100
    # 确保它们实际上是同一个对象，而不只是有相同的值
    print(net[2].weight.data[0] == net[4].weight.data[0])


def main():
    # para1()
    # para2()
    # parameter_initialization()
    # parameter_customization()
    parameter_share()


if __name__ == '__main__':
    main()
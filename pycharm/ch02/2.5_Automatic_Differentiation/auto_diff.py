# -*- coding: utf-8 -*-
# @Time : 2022/9/26 12:56
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : auto_diff.py
# @Project : LiMuAI
import torch


def auto_diff():
    x = torch.arange(4.0, requires_grad=True)
    print(x)
    print(x.grad)

    y = 2 * torch.dot(x, x)
    print(y)

    #调用反向传播函数来自动计算y关于x每个分量的梯度x.grad
    y.backward()
    x.grad
    print(x.grad)

    print(x.grad == 4 * x)

    # 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
    x.grad.zero_()
    y = x.sum()
    y.backward()
    x.grad
    print(x.grad)

def auto_diff_matrix():
    print("------非标量变量的反向传播------")
    #我们的目的不是计算微分矩阵，而是单独计算批量中每个样本的偏导数之和。
    x = torch.arange(4.0, requires_grad=True)
    # 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
    # 在我们的例子中，我们只想求偏导数的和，所以传递一个1的梯度是合适的
    y = x * x
    # 等价于y.backward(torch.ones(len(x)))

    #sum() 方法对序列进行求和计算。
    y.sum().backward()
    print(x.grad)

def separation_calculation():
    print("------分离计算------")
    x = torch.arange(4.0, requires_grad=True)
    y = x * x

    '''
    学习链接：https://bnikolic.co.uk/blog/pytorch-detach.html#:~:text=The%20detach()%20method%20constructs,visualised%20using%20the%20torchviz%20package
    '''
    #detach() 返回一个与当前图分离的新张量。结果永远不需要梯度
    #计算z = u * x对x的偏导数时，将u作为常数处理
    u = y.detach()
    print(u)
    z = u * x
    print(z)
    z.sum().backward()
    print(x.grad == u)

    x.grad.zero_()
    y.sum().backward()
    print(x.grad == 2 * x)

def gradient_calculation():
    def f(a):
        b = a * 2
        while b.norm() < 1000:
            b = b * 2
        if b.sum() > 0:
            c = b
        else:
            c = 100 * b
        return c
    print("---Python控制流不会影响梯度计算---")
    a = torch.randn(size=(), requires_grad=True)
    print(a)
    d = f(a)
    print(d)
    d.backward()

    print(a.grad)
    print(a.grad == d / a)

def main():
    auto_diff()
    auto_diff_matrix()
    separation_calculation()
    gradient_calculation()

if __name__ == '__main__':
    main()
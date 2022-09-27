# -*- coding: utf-8 -*-
# @Time : 2022/9/26 13:56
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : linear_regression.py
# @Project : LiMuAI

'''
----------------------------------------------------------------------
线性回归恰好是一个在整个域中只有一个最小值的学习问题。
但是对于像深度神经网络这样复杂的模型来说，损失平面上通常包含多个最小值。
深度学习实践者很少会去花费大力气寻找这样一组参数，使得在训练集上的损失达到最小。
事实上，更难做到的是找到一组参数，这组参数能够在我们从未见过的数据上实现较低的损失，
这一挑战被称为泛化（generalization）。
----------------------------------------------------------------------
调参（hyperparameter tuning）是选择超参数的过程。
超参数通常是我们根据训练迭代结果来调整的，
而训练迭代结果是在独立的验证数据集（validation dataset）上评估得到的。
----------------------------------------------------------------------
'''

import math
import time
import numpy as np
import torch
from d2l import torch as d2l

class Timer:
    """计算多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

def linear_regression():
    print("------linear gression------")
    n = 10000
    a = torch.ones(n)
    b = torch.ones(n)
    c = torch.zeros(n)
    timer = Timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    print(f'{timer.stop():.5f} sec')

    timer.start()
    d = a + b
    print(f'{timer.stop():.5f} sec')

def norm():
    # 再次使用numpy进行可视化
    x = np.arange(-7, 7, 0.01)

    # 均值和标准差对
    params = [(0, 1), (0, 2), (3, 1)]
    d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
             ylabel='p(x)', figsize=(4.5, 2.5),
             legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
    d2l.plt.show()

def main():
    print("如今在深度学习中的灵感同样或更多地来自数学、统计学和计算机科学。")
    #linear_regression()
    #norm()

if __name__ == '__main__':
    main()
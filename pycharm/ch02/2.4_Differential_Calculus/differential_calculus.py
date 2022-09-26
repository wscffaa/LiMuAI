# -*- coding: utf-8 -*-
# @Time : 2022/9/21 17:45
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : differential_calculus.py
# @Project : LiMuAI

import torch
import numpy as np
from d2l import torch as d2l

def diff_calculius():

    print("优化（optimization）：⽤模型拟合观测数据的过程")

    #x = 1 时，f(x)= -1 ， 该点（1，-1）处的导数为 6x - 4 = 2，斜率为2
    def f(x):
        return 3 * x ** 2 - 4 * x

    def numerical_lim(f, x, h):
        return (f(x + h) - f(x)) / h

    def instantaneous():
        h = 0.1
        for i in range(5):
            print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
            h *= 0.1

    def test_1():
        x = np.arange(0, 3, 0.1)
        d2l.plot(x, [f(x), 2 * x -3], 'x', 'f(x)', legend=['x', 'Tangent line (x=1)'])
        d2l.plt.show();

    # x = 1 时，f(x)= 0 ， 该点（1，0）处的导数为 3x^2 + 1/x^2 = 4，斜率为4
    def test_2():
        x = np.arange(0, 3, 0.1)
        d2l.plot(x, [x ** 3 - 1 / x, 4 * x - 4], 'x', 'f(x)', legend=['x', 'Tangent line (x=1)'])
        d2l.plt.show();

    #不同test对应不同测试内容
    instantaneous()
    #test_1()
    test_2()



def main():
    diff_calculius()

if __name__ == '__main__':
    main()
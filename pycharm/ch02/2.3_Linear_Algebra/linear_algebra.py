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

def tensor():
    print("向量是一阶张量，矩阵是二阶张量")
    X = torch.arange(24).reshape(2, 3, 4)
    print(X)

    print("任何按元素二元运算的结果都将是相同形状的张量")
    A = torch.arange(30, dtype=torch.float32).reshape(3, 2, 5)
    B = A.clone()
    print(B.shape)
    print(B)

    print("两个矩阵的按元素乘法称为Hadamard积")
    print("hadamard积是2个矩阵按照元素相乘，不改变torch.Size，与矩阵乘法不一样")
    print(A * B)

    print("将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘")
    a = 2
    t = A + a
    print(t)
    print(t.shape)

def downscaling():
    print("对张量元素进行求和用sum函数")
    x = torch.arange(4, dtype=torch.float32)
    print(x)
    print(x.sum())
    print("调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量")

    A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    print(A)
    A_sum_axis0 = A.sum(axis=0)
    A_sum_axis1 = A.sum(axis=1)

    print("axis=0指定求和所有行的元素来降维--竖着求，维度为列数")
    print(A_sum_axis0)

    print("axis=1指定求和所有列的元素来降维--横着求，维度为行数")
    print(A_sum_axis1)

    print("通过mean函数将总和除以元素总数来计算平均值")
    print(A.mean())
    print(A.sum()/A.numel())

    print("算平均值的函数也可以沿指定轴降低张量的维度")
    print(A.mean(axis=0))

    print(A.mean(axis=1))

    print("通过keepdims来保持轴数不变")

    print("axis=0, 求和所有行的元素，保持列数不变")
    sum_A_axis0 = A.sum(axis=0, keepdims=True)
    print(sum_A_axis0)

    print("axis=1, 求和所有列的元素，保持行数不变")
    sum_A_axis1 = A.sum(axis=1, keepdims=True)
    print(sum_A_axis1)

    print("通过广播将A除以sum_A")
    print(A / sum_A_axis0)
    print(A / sum_A_axis1)

    print("如果我们想沿某个轴计算A元素的累积总和， 比如axis=0（按行计算），我们可以调用cumsum函数。 此函数不会沿任何轴降低输入张量的维度")
    print(A)
    print(A.cumsum(axis=0))
    print(A.cumsum(axis=1))

def DotProduct():
    print("点积是相同位置的按元素乘积的和")
    x = torch.arange(4, dtype=torch.float32)
    y = torch.ones(4, dtype=torch.float32)
    print(x, y, torch.dot(x, y))

    print("通过执行按元素乘法，然后进行求和来表示两个向量的点积")
    print(torch.sum(x * y))

def MatrixVectorProduct():
    print("矩阵A是m*n的矩阵，x是n*1的列向量，矩阵向量积Ax是一个长度为m的列向量")
    A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    x = torch.arange(4, dtype=torch.float32)
    print(A.size(), x.size())
    print(torch.mv(A, x))

def MatrixMatrixMultiplication():
    print("矩阵AB相乘可以看做是执行m次矩阵向量积，然后将结果拼接在一起形成一个n * m 矩阵")
    A = torch.arange(20, dtype=torch.float32).reshape(5 ,4)
    B = torch.arange(12, dtype=torch.float32).reshape(4 ,3)
    result = torch.mm(A ,B)
    print(A)
    print(B)
    print(result)
    print(result.size())

def norm():
    print("一个向量的范数告诉我们一个向量有多大")
    print("L2范数是向量元素平方和的平方根")
    u = torch.tensor([3.0, -4.0])
    print(torch.norm(u))

    print("L1范数是向量元素的绝对值之和")
    print(torch.abs(u).sum())

    print("robenius范数（Frobenius norm）是矩阵元素平方和的平方根")
    print(torch.norm(torch.ones((4, 9))))
def main():
    print("-----本章主要内容是线性代数-----")
    print("0-d（标量）通常代表一个类别")
    print("1-d（向量）通常代表一个特征向量")
    print("2-d（矩阵）通常代表一个样本，即特征矩阵")
    print("3-d通常代表RGB图片，即CHW")
    print("4-d通常代表一个RGB图片批量，即BCHW")
    print("5-d通常代表一个视频批次，即BTCHW")
    print("------标量------")
    scalar()
    print("------向量------")
    element()
    print("------矩阵------")
    matrices()
    print("------张量------")
    tensor()
    print("------降维------")
    downscaling()
    print("------点积------")
    DotProduct()
    print("------矩阵-向量积------")
    MatrixVectorProduct()
    print("------矩阵-矩阵乘法-----")
    MatrixMatrixMultiplication()
    print("------范数------")
    norm()



if __name__ == '__main__':
    main()
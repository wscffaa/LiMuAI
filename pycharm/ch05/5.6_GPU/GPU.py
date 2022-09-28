# -*- coding: utf-8 -*-
# @Time : 2022/9/28 10:47
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : GPU.py
# @Project : LiMuAI
import torch

def main():
    print("张量是在内存中创建的，然后使用CPU计算它")
    print("cpu设备意味着所有物理CPU和内存")
    print("gpu设备只代表一个卡和相应的显存")

    x = torch.tensor([1, 2, 3])
    print(x.device)

    print("只要所有的数据和参数都在同一个设备上， 我们就可以有效地学习模型")


if __name__ == '__main__':
    main()
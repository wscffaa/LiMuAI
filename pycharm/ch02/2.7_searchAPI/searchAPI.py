# -*- coding: utf-8 -*-
# @Time : 2022/9/26 13:48
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : searchAPI.py
# @Project : LiMuAI
import torch
def main():
    print("为了知道模块中可以调用哪些函数和类，我们调用dir函数")
    print(dir(torch.distributions))
    help(torch.arange)


if __name__ == '__main__':
    main()
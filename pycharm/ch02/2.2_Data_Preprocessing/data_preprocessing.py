#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2022/9/15 上午11:24
@Author  : feifan
@FileName: data_preprocessing.py
@Software: PyCharm
"""

import torch
import os
import pandas as pd

def data_prepare():
    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    data_file = os.path.join('..', 'data', 'house_tiny.csv')
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')
        f.write('NA,Pave,127500\n')
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')
    data = pd.read_csv(data_file)
    print(data)

    return data

def data_NaN(data):
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs = inputs.fillna(inputs.mean())
    print(inputs)

    print("对于inputs中的类别值或离散值，我们将“NaN”视为一个类别")
    print("pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”")
    inputs = pd.get_dummies(inputs, dummy_na=True)
    print(inputs)

    return inputs, outputs

def data_toTensor(inputs, outputs):
    print("-------inputs-------")
    print(inputs)
    print("-------outputs-------")
    print(outputs)

    X, Y = torch.tensor(inputs.values), torch.tensor(outputs.values)

    print("-------inputs_Tensor-------")
    print(X)
    print("-------outputs_Tensor-------")
    print(Y)

def main():
    data = data_prepare()
    inputs, outputs = data_NaN(data)
    data_toTensor(inputs, outputs)


if __name__ == '__main__':
    main()
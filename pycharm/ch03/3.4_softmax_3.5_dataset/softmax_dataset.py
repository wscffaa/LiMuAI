# -*- coding: utf-8 -*-
# @Time : 2022/9/26 17:47
# @Author : feifan
# @Email : feifan@shu.edu.cn
# @File : softmax_dataset.py
# @Project : LiMuAI
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

def get_dataset():
    """通过ToTensor实例讲图像数据从PIL类型转换为32位浮点数格式"""
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return mnist_train, mnist_test

def get_fashion_mnist_labels(labels):
    """返回数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes

def get_dataloader_workers():
    return 4

def dataset():
    mnist_train, mnist_test = get_dataset()
    print(len(mnist_train), len(mnist_test))

    print(mnist_train[0][0].shape)

    #展示数据集label内容
    # X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    # show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

    batch_size = 256

    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                 num_workers=get_dataloader_workers())

    timer = d2l.Timer()
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')

def load_data_fashion_mnist(batch_size, resize=None):
    """下载数据集并加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)

    train_dataloader = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                       num_workers=get_dataloader_workers())

    test_dataloader = data.DataLoader(mnist_test, batch_size, shuffle=True,
                                      num_workers=get_dataloader_workers())

    return (train_dataloader, test_dataloader)


def main():
    print("------softmax回归任然是一个线性模型------")
    print("softmax运算获取一个向量并将其映射为概率")
    print("softmax回归适用于分类问题，它使用了softmax运算中输出类别的概率分布")

    #dataset()
    train_iter,test_iter = load_data_fashion_mnist(32, resize=64)
    for X,y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break



if __name__ == '__main__':
    main()
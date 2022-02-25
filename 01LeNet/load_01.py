import gzip
import os

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# 重写torch.utils.data的Dataset中的方法
class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, folder, data_name, label_name, transform=None):
        # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        (train_set, train_labels) = self.load_data(folder, data_name, label_name)
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):

        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)
    '''
    load_data也是我们自定义的函数，用途：读取数据集中的数据 ( 图片数据+标签label
    '''
    def load_data(self, data_folder, data_name, label_name):
        with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
            x_train = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
        return x_train, y_train


trainDataset = DealDataset('../Datasets/mnist_dataset', "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                           transform=transforms.ToTensor())

# 训练数据和测试数据的装载
train_loader = torch.utils.data.DataLoader(
    dataset=trainDataset,
    batch_size=10,  # 一个批次可以认为是一个包，每个包中含有10张图片
    shuffle=False,
)

# 实现单张图片可视化
images, labels = next(iter(train_loader))
img = torchvision.utils.make_grid(images)

img = img.numpy().transpose(1, 2, 0)
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
print(labels)
plt.imshow(img)
plt.show()

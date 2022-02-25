import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('..')
import d2lzh_pytorch as d2l
from torch.utils.data import DataLoader

# mnist_train = torchvision.datasets.MNIST(root='../Datasets/mnist_dataset/', train=True, download=False,
#                                          transform=transforms.ToTensor())
# mnist_test = torchvision.datasets.MNIST(root='../Datasets/mnist_dataset/', train=False, download=False,
#                                         transform=transforms.ToTensor())
#
# print(type(mnist_train))
# print(len(mnist_train), len(mnist_test))


def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])]
)

train_dataset = torchvision.datasets.FashionMNIST(root='../Datasets/', train=True, transform=data_tf,
                                                  download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='../Datasets/', train=False, transform=data_tf,
                                                 download=True)

print(type(train_dataset))
print(len(train_dataset), len(train_dataset))

# 访问任意一个样本
feature, label = train_dataset[0]
print(feature.shape, label)

# 展示数据集中前十个数据图片
X, y = [], []
for i in range(10):
    X.append(train_dataset[i][0])
    y.append(train_dataset[i][1])
show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))


batch_size = 256
# Pytorch的DataLoader是允许使用多进程来加速数据读取
# num_workers 读取数据进程的数量
# if sys.platform.startswith('win'):
#     num_workers = 0
#     print('num_workers:', num_workers)
# else:
#     num_workers = 4
#     print('num_workers:', num_workers)
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 获取并读取Fashion-MNIST数据集的逻辑已经封装在d2lzh_pytorch.load_data_fashion_mnist函数中

# 读取一遍训练数据需要的时间
start = time.time()
for x, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))

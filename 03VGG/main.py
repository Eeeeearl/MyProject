import time
import torch
from torch import nn, optim
import torchvision

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 使得宽高减半

    return nn.Sequential(*blk)


def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels,) in enumerate(conv_arch):
        # 每经过一个block都会宽高减半
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    net.add_module("fc",
                   nn.Sequential(
                       d2l.FlattenLayer(),
                       nn.Linear(fc_features, fc_hidden_units),
                       nn.ReLU(),
                       nn.Dropout(0.2),
                       nn.Linear(fc_hidden_units, fc_hidden_units),
                       nn.ReLU(),
                       nn.Dropout(0.5),
                       nn.Linear(fc_hidden_units, 10)
                   ))
    return net


# 定义超参数 卷积层个数 + 输入通道数 + 输出通道数
conv_arch = ((2, 1, 64), (2, 64, 128), (3, 128, 256), (3, 256, 512), (3, 512, 512))
# 经过5个vgg_block，宽高会减半5次，变成 224/32 = 7
fc_features = 512 * 7 * 7  # c * w * h
fc_hidden_units = 4096


net = vgg(conv_arch, fc_features, fc_hidden_units)
print(net)
X = torch.rand(1, 1, 224, 224)

for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape:', X.shape)

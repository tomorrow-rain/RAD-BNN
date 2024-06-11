import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import *

__all__ = ['resnet20_1w1a']

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_1w1a, self).__init__()

        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(planes)
        self.prelu2 = nn.PReLU(planes)

        self.shortcut = nn.Sequential()
        pad = 0 if planes == self.expansion*in_planes else planes // 4
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                        nn.AvgPool2d((2, 2)),
                        LambdaLayer(lambda x:
                        F.pad(x, (0, 0, 0, 0, pad, pad), "constant", 0)))

        self.dada1 = nn.Sequential(
            nn.AvgPool2d((8, 8)),
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid()
        )
        self.dada2 = nn.Sequential(
            nn.AvgPool2d((8, 8)),
            nn.Conv2d(planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv1(x)
        out *= self.dada1(x)
        out = self.prelu1(out)
        out = self.bn1(out)
        out += self.shortcut(x)

        x1 = out
        out = self.conv2(out)
        out *= self.dada2(x1)
        out = self.prelu2(out)
        out = self.bn2(out)
        out += x1
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu = nn.PReLU(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        return out


def resnet20_1w1a(num_classes=10):
    return ResNet(BasicBlock_1w1a, [3, 3, 3],num_classes=num_classes)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda name: 'conv' in name or 'linear' in name, [name[0] for name in list(net.named_modules())]))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
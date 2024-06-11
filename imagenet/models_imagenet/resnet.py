import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
from modules import *
import torch.nn.functional as F

BN = None

__all__ = ['resnet18_1w1a', 'resnet34_1w1a']


def conv3x3Binary(in_planes, out_planes, stride=1):
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3Binary(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.nonlinear1 = nn.PReLU(planes)
        self.conv2 = conv3x3Binary(planes, planes)
        self.bn2 = BN(planes)
        self.nonlinear2 = nn.PReLU(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.AvgPool2d((2, 2)),
                nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes)
            )

        self.dada1 = nn.Sequential(
            nn.AvgPool2d((7, 7)),
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid()
        )
        self.dada2 = nn.Sequential(
            nn.AvgPool2d((7, 7)),
            nn.Conv2d(planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv1(x)
        out *= self.dada1(x)
        out = self.nonlinear1(out)
        out = self.bn1(out)
        out += self.downsample(x)

        residual = out
        out = self.conv2(out)
        out *= self.dada2(residual)
        out = self.nonlinear2(out)
        out = self.bn2(out)
        out += residual
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, 
                 bypass_last_bn=False,
                 bn_group_size=1,
                 bn_group=None,
                 bn_sync_stats=False,
                 use_sync_bn=True):

        global BN

        BN = nn.BatchNorm2d

        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(64)
        self.nonlinear = nn.PReLU(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1e-8)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.maxpool(x)     

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.fc(x)

        return x


def resnet18_1w1a(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_1w1a(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda name: 'conv' in name or 'fc' in name, [name[0] for name in list(net.named_modules())]))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
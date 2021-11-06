import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D
import kymatio.datasets as scattering_datasets
import argparse
import math
from torchsummary import summary
import os
import sys

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)
"""
def conv1x1(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=True)
"""
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def output_grad(self):
        print('conv0:',self.conv0.weight.grad)
        print('conv1:',self.conv1.weight.grad)
        print('conv2:',self.conv2.weight.grad)
        print('conv3:',self.conv3.weight.grad)
        sys.stdout.flush()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class scatternet_cnn(nn.Module):
    def __init__(self,J=3,N=224, num_classes=1000):
        super(scatternet_cnn, self).__init__()

        print(J)
        self.scattering = Scattering2D(J=3, L=8, shape=(224, 224))
        self.nspace = int(N / (2 ** J))
        self.nfscat = int(1 + 8 * J + 8 * 8 * J * (J - 1) / 2)
        self.ichannels = 256
        self.ichannels2 = 512
        self.inplanes = self.ichannels
        print('nfscat',self.nfscat)
        print('nspace',self.nspace)

        self.bn0 = nn.BatchNorm2d(int(3 * self.nfscat), eps=1e-5, momentum=0.9, affine=False)

        self.conv1 = nn.Conv2d(int(3*self.nfscat), self.ichannels, kernel_size=3,padding=1)#zconv3x3_3D(self.nfscat,self.ichannels)
        self.bn1 = nn.BatchNorm2d(self.ichannels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, self.ichannels, 2)
        self.layer2 = self._make_layer(BasicBlock, self.ichannels2, 2, stride=2)
        self.avgpool = nn.AvgPool2d(int(self.nspace/2))
        self.fc = nn.Linear(self.ichannels2, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if(m.affine):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes ,
                          kernel_size=1, stride=stride, bias=True)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.scattering(x)
        x = x.view(x.size(0), 3 * self.nfscat, self.nspace, self.nspace)
        x = self.bn0(x)
        x = self.conv1(x)
        x = x.view(x.size(0), self.ichannels, self.nspace, self.nspace)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.Sigmoid()(x)
        return x

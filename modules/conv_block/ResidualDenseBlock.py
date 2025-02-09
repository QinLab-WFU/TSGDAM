# -*- coding:utf-8 -*-
# author:yuwanmo
# contact: a13767092838@gmail.com
# datetime:2022/5/15 22:36
import torch
import torch.nn as nn
import torch.nn.functional as F


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer=3, growthRate=32):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)

        out = out + x
        return out


# modify

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, is_batchnorm, kernel_size=3):
        super(make_dense, self).__init__()
        if is_batchnorm:
            self.conv = nn.Sequential(nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False),
                                      nn.BatchNorm2d(growthRate),
                                      nn.ReLU(inplace=True)
                                      )
        else:
            self.conv = nn.Sequential(nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False),
                                      nn.ReLU(inplace=True)
                                      )

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, in_size, is_batchnorm, nDenselayer=3, growthRate=32):
        super(RDB, self).__init__()
        nChannels_ = in_size
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, is_batchnorm, ))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, in_size, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)

        out = out + x
        return out

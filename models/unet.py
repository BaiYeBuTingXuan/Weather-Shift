# Auther:Hejun WANG
# -*- coding: utf-8 -*-
import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import torch.nn as nn
import torch
from torch.autograd import Variable
from models import PADCell


class UNetDown(nn.Module):
    """
        UNetDown for image process
        input:x
        output:encoded x
        structure:Conv2d-->Norm(if chosen)-->LeakRelu-->Dropout(if chosen)
        padding: circular pad on left and right, zero pad on top and bottom
    """
    def __init__(self, in_size, out_size, kernel_size=4, stride=2, padding=1, normalize=True, leaky=0.2, dropout=0.5):
        super(UNetDown, self).__init__()
        layers = [PADCell(padding=padding),
                  nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, bias=False)]

        if normalize:   # 是否归一化
            layers.append(nn.InstanceNorm2d(out_size))

        layers.append(nn.LeakyReLU(leaky))    # 激活函数层

        if dropout:   # 是否Dropout
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """
        UNetUp for image process
        input:x,skip_input
        output:model(x) cat skip_input
        structure:Conv2d-->Norm-->Relu-->Dropout(if chosen)
    """
    def __init__(self, in_size, out_size, kernel_size=2, stride=2, normalize=True, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
                  nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride, bias=False),
                  nn.InstanceNorm2d(out_size),
                  nn.ReLU(inplace=True)]
        
        
        if normalize:   # 是否归一化
            layers.append(nn.InstanceNorm2d(out_size))

        layers.append(nn.ReLU(inplace=True))    # 激活函数层
        
        if dropout:  # 是否Dropout
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)

        return torch.cat((x, skip_input), 1)
    

class UNetGenerator(nn.Module):
    """
        Total Network from Image、Route to Path Graph
        Input(RGB image V, Local Route R)-->UNet-->Path Graph P
        ----------------------------------------------------------------------------
        Input:
        :param V,R:4D torch.tensor:4D torch.tensor(batch_size * 2RGB * Width * Height)
        ----------------------------------------------------------------------------
        Output:
        :return P:Gray Picture of extraction from path,in order to create CostMap
    """
    def __init__(self, in_channels=3, out_channels=3, weather_num=10, weather_embedding=16):
        super(UNetGenerator, self).__init__()
        # 定义encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # 定义decoder
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        # 定义输出
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            PADCell(padding=(1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.ReLU()
        )

        def embeding_block(in_size, out_size):
            layer = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, bias=False),
                                  nn.Tanh()
                                 )
            return layer
        
        self.embedding1 = embeding_block(weather_num, 64)
        self.embedding2 = embeding_block(64, 128)
        self.embedding3 = embeding_block(128, 256)
        self.embedding4 = embeding_block(256, 512)
        self.embedding5 = embeding_block(512, 512)

    def weather_embedding(self, x):
        o1 = self.embedding1(x)
        o2 = self.embedding2(o1)
        o3 = self.embedding3(o2)
        o4 = self.embedding4(o3)
        o5 = self.embedding5(o4)
        return o1, o2, o3, o4, o5

    def merge(self, res, x):
        x[:, 2, :, :] = nn.ReLU()(x[:, 2, :, :] - res[:, 2, :, :])

        weight = torch.stack([x[:, 2, :, :], x[:, 2, :, :]], dim=1)

        x[:, 0:2, :, :] = (x[:, 0:2, :, :] * weight + res[:, 0:2, :, :])/(weight+1)
        
        return x

    def forward(self, x, w):
        '''
        w batch_size*10
        x batch_size*128*256
        '''
        w = w.unsqueeze(2).unsqueeze(3)
        w1, w2, w3, w4, w5 = self.weather_embedding(w)

        d1 = self.down1(x) + w1
        d2 = self.down2(d1) + w2
        d3 = self.down3(d2) + w3
        d4 = self.down4(d3) + w4
        d5 = self.down5(d4) + w5

        d6 = self.down6(d5)

        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        u6 = self.final(u5)

        result = self.merge(u6,x)

        return result
    

if __name__ == '__main__':
    g = UNetGenerator()
    img = torch.rand([32, 3, 128, 256])
    w = torch.rand([32, 10])
    img = g(img,w)

    print(img.size())
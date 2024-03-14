# Auther:Hejun WANG
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
from torch.autograd import Variable

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
        layers = [nn.CircularPad2d(padding=(padding,padding,0,0)),
                  nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=(padding,0), bias=False)]

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
    def __init__(self, in_size, out_size, kernel_size=4, stride=2, padding=1, normalize=True, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.CircularPad2d(padding=(padding,padding,0,0)),
                  nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=(padding,0), bias=False),
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
    

class GeneratorUNet(nn.Module):
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
    def __init__(self, in_channels=6, out_channels=3):
        super(GeneratorUNet, self).__init__()
        # 定义encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # 定义decoder
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        # 定义输出
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.CircularPad2d((1, 0, 0, 0)),
            nn.ZeroPad2d((0, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.ReLU()
        )

    def merge(self, res, x):
        x[:, 2, :, :] = nn.ReLU(x[:, 2, :, :] - res[:, 2, :, :])

        weight = torch.stack([x[:, 2, :, :], x[:, 2, :, :]], dim=1)
        x[:, 0:1, :, :] = (x[:, 0:1, :, :] * weight + res[:, 0:1, :, :])/(weight+1)
        
        return x

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        res = self.final(u6)

        result = self.merge(res,x)

        return result
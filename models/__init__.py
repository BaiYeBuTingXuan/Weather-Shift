# Auther:Hejun WANG
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from torch.autograd import Variable


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def custom_padding(input_tensor, left_padding, right_padding, top_padding, bottom_padding):
    batch_size, channels, height, width = input_tensor.size()
    
    # 左边填充
    left_pad = input_tensor[:, :, :, -left_padding:]
    # 右边填充
    right_pad = input_tensor[:, :, :, :right_padding]
    # 上边填充
    top_pad = torch.zeros(batch_size, channels, top_padding, width+left_padding+right_padding, device=input_tensor.device)
    # 下边填充
    bottom_pad = torch.zeros(batch_size, channels, bottom_padding, width+left_padding+right_padding, device=input_tensor.device)
    
    # 拼接填充后的特征图
    padded_tensor = torch.cat((left_pad, input_tensor, right_pad), dim=3)
    padded_tensor = torch.cat((top_pad, padded_tensor, bottom_pad), dim=2)
    
    return padded_tensor


class CNNCell(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride, padding, bias=False):
        layers = [nn.CircularPad2d(padding=(padding,padding,0,0)),
                  nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=(padding,0), bias=False)]


class LSTMCell(nn.Module):
    """
        LSTMCell for image process
        input:x
        output:model(x)
        structure:Input-->LSTM-->Output
    """
    def __init__(self, input_channels, hidden_channels, cell_num=1, layers_num=4, dropout=0.5, bias=True):
        super(LSTMCell, self).__init__()
        self.layers = nn.ModuleList([torch.nn.LSTM(input_size=input_channels, hidden_size=hidden_channels, num_layers=layers_num, bias=bias, batch_first=True, dropout=dropout)
                                     for _ in range(cell_num)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class PADCell(nn.Module):
    def __init__(self, padding=1):
        super(PADCell, self).__init__()
        self.is_int = type(padding) == int
        self.padding = padding

    def forward(self, x):
        if self.is_int:
            x = custom_padding(x, left_padding=self.padding, right_padding=self.padding, top_padding=self.padding, bottom_padding=self.padding)
        else:
            x = custom_padding(x, left_padding=self.padding[0], right_padding=self.padding[1], top_padding=self.padding[2], bottom_padding=self.padding[3])
        return x


class WeatherClassifier(nn.Module):
    def __init__(self, in_channels=3, out_channel=9+1):
        super(WeatherClassifier, self).__init__()

        # 定义Discriminator的单元结构
        def discriminator_block(in_size, out_size, kernel_size=4, stride=2, padding=1, normalize=True, leaky=0.2, dropout=0.2, pooling=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [PADCell()]
            layers.append(nn.Conv2d(in_size, out_size, kernel_size= kernel_size, stride=stride))
            if normalize:
                layers.append(nn.InstanceNorm2d(out_size))
            layers.append(nn.LeakyReLU(leaky, inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))
            if pooling:
                layers.append(nn.MaxPool2d(kernel_size=4, stride=2, padding=1))
            return layers
        

        self.feature_extracter = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False, pooling=False),
            *discriminator_block(64, 128, pooling=False),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            # nn.ZeroPad2d((0,0,1,0)),
            nn.Conv2d(512, 256, kernel_size=(2,4), stride=1, padding=0, bias=False),
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
            nn.Conv2d(256, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.size())
        x = self.feature_extracter(x)
        batch_size, _, _, _ = x.size()
        x = x.reshape(batch_size, -1)
        x = self.activation(x)
        return x


if __name__ == '__main__':
    img = torch.rand([32,3,128,256])
    d = WeatherClassifier()
    x = d(img)
    print(x.size())
    print("good！")

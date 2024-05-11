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
from models.activation import RoundActivation, StepActivation

class Atttention(nn.Module):
    def __init__(self, in_size, out_size, feature_size=256, kernel_size=4, stride=2, padding=1, normalize=True, leaky=0.2, dropout=0.5):
        super(Atttention, self).__init__()
        
        self.query = nn.Sequential(*self.get_layers(in_size, feature_size))
        self.key = nn.Sequential(*self.get_layers(in_size, feature_size))
        self.value = nn.Sequential(*self.get_layers(in_size, feature_size))


    def get_layers(in_size, feature_size):
        layers = [nn.Conv2d(in_size, feature_size, kernel_size=1, stride=1, bias=False),
                  nn.Tanh(),
                  nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, bias=False),
                  nn.InstanceNorm(),
                  nn.Tanh(),
                  ]
        return layers

    def forward(self, x):
        q = self.query(x)
        k = self.key(x).transpose(1,2)
        v = self.value(x)
        a = torch.softmax(torch.matmul(q*k), dim=2)
    
        return a*v
    
class Patch(nn.Module):
    def __init__(self, in_size, out_size, scale=100):
        super(Patch, self).__init__()
        self.conv1 = nn.Sequential(*self.get_layers(in_size, 256))
        self.atten1 = Atttention(in_size=256,feature_size=256)

        self.conv2 = nn.Sequential(*self.get_layers(256, 256))
        self.atten2 = Atttention(in_size=256,feature_size=256)

        self.conv3 = nn.Sequential(*self.get_layers(256, 256))
        self.atten3 = Atttention(in_size=256,feature_size=256)

        self.final = [
            nn.Conv2d(256, 128, kernel_size=1, stride=1,padding = 1, bias=False),
            nn.Tanh(),
            nn.Conv2d(256, 128, kernel_size=1, stride=1,padding = 1, bias=False),
            nn.Tanh(),
            nn.Conv2d(256, out_size, kernel_size=1, stride=1,padding = 1, bias=False),
            nn.Sigmoid()
        ]

        self.final = nn.Sequential(*self.final)

        self.scale = scale

    def get_layers(in_size, feature_size):
        layers = [nn.Conv2d(in_size, feature_size, kernel_size=1, stride=1, bias=False),
                  nn.Relu(),
                  nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, bias=False),
                  ]
        return layers
    
    def forward(self,x):
        x = self.conv1(x)
        x = x + self.atten1(x)

        x = self.conv2(x)
        x = x + self.atten2(x)

        x = self.conv3(x)
        x = x + self.atten3(x)

        x = self.final(x)*self.scale

        return x
    
class Patch_Discriminator(nn.Module):
    def __init__(self, in_size, out_size):
        super(Patch, self).__init__()
        self.conv1 = nn.Sequential(*self.get_layers(in_size, 256))
        self.atten1 = Atttention(in_size=256,feature_size=256)

        self.conv2 = nn.Sequential(*self.get_layers(256, 128))
        self.atten2 = Atttention(in_size=128,feature_size=128)

        self.conv3 = nn.Sequential(*self.get_layers(128, 128))
        self.atten3 = Atttention(in_size=128,feature_size=128)

        self.final = [
            nn.Conv2d(128, 32, kernel_size=1, stride=1,padding = 1, bias=False),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1,padding = 1, bias=False),
            nn.Tanh(),
            nn.Conv2d(32, out_size, kernel_size=1, stride=1,padding = 1, bias=False),
            nn.Softmax()
        ]

        self.final = nn.Sequential(*self.final)


    def get_layers(in_size, feature_size):
        layers = [nn.Conv2d(in_size, feature_size, kernel_size=1, stride=1, bias=False),
                  nn.Relu(),
                  nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, bias=False)]
        return layers
    
    def forward(self,x):
        x = self.conv1(x)
        x = x + self.atten1(x)

        x = self.conv2(x)
        x = x + self.atten2(x)

        x = self.conv3(x)
        x = x + self.atten3(x)

        x = self.final(x)

        return x

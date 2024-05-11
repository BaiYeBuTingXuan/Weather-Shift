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

class InvExp_Loss(nn.Module):
    def __init__(self, sigma=1):
        super(InvExp_Loss, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        # loss = torch.sum(torch.exp(-x**2/(self.sigma*self.sigma)))
        loss = torch.mean(torch.exp(-torch.abs(x)/self.sigma))

        return loss
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
    
class UNet(nn.Module):
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
    def __init__(self, in_channels=3, out_channels=3, weather_num=10):
        super(UNet, self).__init__()


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
            # PADCell(padding=(1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size=1, stride=1),
            nn.Tanh(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=True),
        )


        def embeding_block(in_size, out_size):
            layer = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, bias=True),
                                  nn.Tanh()
                                 )
            return layer
        
        self.embedding1 = embeding_block(weather_num, 64)
        self.embedding2 = embeding_block(64, 128)
        self.embedding3 = embeding_block(128, 256)
        self.embedding4 = embeding_block(256, 512)
        self.embedding5 = embeding_block(512, 512)

        self.round = RoundActivation()
        self.step = StepActivation()

    def weather_embedding(self, x):
        o1 = self.embedding1(x)
        o2 = self.embedding2(o1)
        o3 = self.embedding3(o2)
        o4 = self.embedding4(o3)
        o5 = self.embedding5(o4)
        return o1, o2, o3, o4, o5

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

        result = self.final(u5)

        return result


class UNetGenerator_Res(nn.Module):
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
    def __init__(self, in_channels=3, out_channels=3, weather_num=10):
        super(UNetGenerator_Res, self).__init__()

        self.normalize = nn.BatchNorm2d(num_features=3)

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
            # PADCell(padding=(1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size=1, stride=1),
            nn.Tanh(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=True),
        )


        def embeding_block(in_size, out_size):
            layer = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, bias=True),
                                  nn.Tanh()
                                 )
            return layer
        
        self.embedding1 = embeding_block(weather_num, 64)
        self.embedding2 = embeding_block(64, 128)
        self.embedding3 = embeding_block(128, 256)
        self.embedding4 = embeding_block(256, 512)
        self.embedding5 = embeding_block(512, 512)

        self.round = RoundActivation()
        self.step = StepActivation()

    def weather_embedding(self, x):
        o1 = self.embedding1(x)
        o2 = self.embedding2(o1)
        o3 = self.embedding3(o2)
        o4 = self.embedding4(o3)
        o5 = self.embedding5(o4)
        return o1, o2, o3, o4, o5

    def forward(self, x, w):
        '''
        w batch_size*10
        x batch_size*128*256
        '''
        w = w.unsqueeze(2).unsqueeze(3)
        w1, w2, w3, w4, w5 = self.weather_embedding(w)

        d0 = self.normalize(x) 
    
        d1 = self.down1(d0) + w1
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

        result = x + u6

        return result

class UNetGenerator_Merge(nn.Module):
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
    def __init__(self, in_channels=3, out_channels=3, weather_num=10):
        super(UNetGenerator_Merge, self).__init__()

        self.normalize = nn.BatchNorm2d(num_features=in_channels)
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels, weather_num=weather_num)

        
        self.step = StepActivation()
        self.round = RoundActivation()


    def merge(self, res, x, epsilon=1e-3):
        z = x.detach().clone()

        particle = self.step(res[:, 2, :, :]) # Is there any particle of weather element
        blocked = self.round(res[:, 2, :, :].clone()) # How many existing point have been blocked
        # print(x[:, 2, :, :].size())
        # print(res[:, 2, :, :].size())
        # print(blocked.size())
        # particle = res[:, 2, :, :] # Is there any particle of weather element
        # blocked = res[:, 2, :, :]

        z[:, 2, :, :] = nn.ReLU()(x[:, 2, :, :] - blocked)

        # weight_x = torch.stack([x[:, 2, :, :], x[:, 2, :, :]], dim=1)
        # weight_p = torch.stack([particle, particle], dim=1)

        z[:, 0, :, :] = (x[:, 0, :, :] * x[:, 2, :, : ]+ res[:, 0, :, :] * particle) / (nn.ReLU()(x[:, 2, :, :] + particle) + epsilon)
        z[:, 1, :, :] = (x[:, 1, :, :] * x[:, 2, :, :] + res[:, 1, :, :] * particle) / (nn.ReLU()(x[:, 2, :, :] + particle) + epsilon)
        z[:, 2, :, :] = nn.ReLU()(x[:, 2, :, :] + particle)
        
        return z

    def forward(self, x, w):
        '''
        w batch_size*10
        x batch_size*128*256
        '''
        x = self.normalize(x) 
    
        z = self.unet(x,w)

        result = self.merge(z,x)

        return result
    
class UNetGenerator_Normal(nn.Module):
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
    def __init__(self, in_channels=5, out_channels=6, weather_num=10, scale=128):
        super(UNetGenerator_Normal, self).__init__()

        self.normalize = nn.BatchNorm2d(num_features=in_channels)
        self.scale = scale
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels, weather_num=weather_num)

    def activate(self, r):
        #         0              1               2          3                4   5
        # r = [mean_radius, mean_reflectance, var_radius, var_reflectance, alpha]

        alpha = nn.Sigmoid()(r[:,4,:,:])
        radius = nn.ReLU()(r[:,[0,2],:,:])*self.scale
        reflectance = nn.Tanh()(r[:,[1,3],:,:])*self.scale

        
        return alpha,radius,reflectance

    def merge(self, x, alpha, radius, reflectance):
        #         0              1               2          3                4 
        # x = [mean_radius, mean_reflectance, var_radius, var_reflectance, rho]
        #                        0              1              
        # radius      =  [mean_radius, var_radius]
        # reflectance = [mean_reflectance, var_reflectance ]


        mean_radius =  radius[:,0,:,:]*alpha + x[:,0,:,:]*(1-alpha)
        mean_reflectance = x[:,1,:,:] + reflectance[:,0,:,:]

        var_radius = alpha*radius[:,1,:,:] + (1-alpha)*x[:,2,:,:] + 2*alpha*(1-alpha)*(radius[:,0,:,:]-x[:,0,:,:])*(radius[:,0,:,:]-x[:,0,:,:])
        var_reflectance = x[:,3,:,:] + reflectance[:,1,:,:]
        # rho = rho

        z = torch.cat((mean_radius.unsqueeze(1), mean_reflectance.unsqueeze(1), var_radius.unsqueeze(1), var_reflectance.unsqueeze(1)), dim=1)
        
        return z

    def forward(self, x, w):
        '''
        w batch_size*10
        x batch_size*128*256
        '''
        # x = x[:,1:6,:,:]
        x = self.normalize(x)
        r = self.unet(x,w)

        alpha,radius,reflectance = self.activate(r)

        x = self.merge(x,alpha,radius,reflectance)

        return x,(alpha,radius,reflectance)
    

class UNetGenerator_Normal2(nn.Module):
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
    def __init__(self, in_channels=5, out_channels=6, weather_num=10, scale=100):
        super(UNetGenerator_Normal2, self).__init__()

        self.normalize = nn.BatchNorm2d(num_features=in_channels)
        self.scale = scale
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels, weather_num=weather_num)

    def activate(self, r):
        #         0              1               2          3                4   5
        # r = [mean_radius, mean_reflectance, var_radius, var_reflectance, alpha]

        alpha = nn.Sigmoid()(r[:,4,:,:])/self.scale
        radius = nn.Sigmoid()(r[:,[0,2],:,:])*self.scale
        reflectance = nn.Tanh()(r[:,[1,3],:,:])*self.scale

        
        return alpha,radius,reflectance

    def merge(self, x, alpha, radius, reflectance):
        #         0              1               2          3                4 
        # x = [mean_radius, mean_reflectance, var_radius, var_reflectance, rho]
        #                        0              1              
        # radius      =  [mean_radius, var_radius]
        # reflectance = [mean_reflectance, var_reflectance ]


        mean_radius =  radius[:,0,:,:]*alpha + x[:,0,:,:]*(1-alpha)
        mean_reflectance = x[:,1,:,:] + reflectance[:,0,:,:]

        var_radius = alpha*radius[:,1,:,:] + (1-alpha)*x[:,2,:,:] + alpha*(1-alpha)*(radius[:,0,:,:]-x[:,0,:,:])*(radius[:,0,:,:]-x[:,0,:,:])
        var_reflectance = x[:,3,:,:] + reflectance[:,1,:,:]
        # rho = rho

        z = torch.cat((mean_radius.unsqueeze(1), mean_reflectance.unsqueeze(1), var_radius.unsqueeze(1), var_reflectance.unsqueeze(1)), dim=1)
        
        return z

    def forward(self, x, w):
        '''
        w batch_size*10
        x batch_size*128*256
        '''
        # x = x[:,1:6,:,:]
        x = self.normalize(x)
        r = self.unet(x,w)

        alpha,radius,reflectance = self.activate(r)

        x = self.merge(x,alpha,radius,reflectance)

        return x,(alpha,radius,reflectance)
    

if __name__ == '__main__':
    g =  UNetGenerator_Normal()
    img = torch.rand([32, 5, 64, 128])
    w = torch.rand([32, 10])
    r,x = g(img,w)
    alpha,rho,radius,reflectance = x
    # img.requires_grad = True
    # w.requires_grad = True

    print(r.size())
    print(len(x))

    
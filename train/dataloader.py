
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
import random
import numpy as np
from PIL import Image
import time
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import copy
from scipy.special import comb
from scipy.stats import beta
np.set_printoptions(suppress=True, precision=4, linewidth=65535)
import matplotlib.pyplot as plt

from utils import print_warning
# from utils.math import angle_normal,spin,sign
from utils.point_cloud import read_pcd

from pathlib import Path
import itertools

# parser = argparse.ArgumentParser()
# parser.add_argument('--test_split', type=float, default=0.1, help='rate of test file')
# parser.add_argument('--valid_split', type=float, default=0.1, help='rate of validation file')
# parser.add_argument('--random_seed', type=int, default=6, help='random seed')
# arg = parser.parse_args()

WEATHERS = ['clear_day' ,'clear_night' ,'dense_fog_day', 'dense_fog_night', 'light_fog_day', 'light_fog_night', 'rain', 'snow_day', 'snow_night', 'None']

def collate_fn(data):
    # print(collate_fn)
    data.sort(key=lambda x: len(x[0]), reverse=True)  # 按照数据长度降序排序
    data_list = []
    lengths = []
    for d in data:
        data_list.append(d[0])
        lengths.append(len(d[0]))
    padded_data = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True)  # 对数据进行填充
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)  # 创建长度张量
    return padded_data, lengths_tensor


class SeeingThroughFogDataset(Dataset):
    def __init__(self, 
                dataset_path='I:\Datasets\DENSE\SeeingThroughFog', 
                splits_path=r'./splits', 
                mode = 'train',
                globe_height = 64,
                globe_width = 128,
                weathers = ['clear_day' ,'clear_night' ,'dense_fog_day', 'dense_fog_night', 'light_fog_day', 'light_fog_night', 'rain', 'snow_day', 'snow_night'],
                lidars = ['lidar_hdl64_strongest']): #  'lidar_vlp32_strongest'
        # self.data_index = data_index
        self.mode = mode
        self.dataset_path = dataset_path
        self.weathers = weathers
        self.lidars = lidars
        self.mode = mode
        self.height = globe_height
        self.width = globe_width


        self.weather_categories = len(weathers)

        globe_transforms = [
                transforms.ToTensor(),
            ]
        if mode == 'train':
            globe_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
        else: # Test or Validation
            pass
        
        self.globe_transforms = transforms.Compose(globe_transforms)

        self.npy_list = {} # self.bin_list[one of lidars][one of weathers] = one of names of globe in *.npy
        for lidar,weather in itertools.product(lidars, weathers):
            if lidar not in self.npy_list.keys():
                self.npy_list[lidar] = {}
            if weather == 'None':
                self.npy_list[lidar]['None'] = []
            else:
                with open(Path(splits_path).joinpath(mode).joinpath(weather+'.txt')) as f:
                    self.npy_list[lidar][weather] = [line.strip().replace(',', '_') for line in f.readlines()]

    def read_cloud(self, path):
        cloud = read_pcd(path)
        return cloud

    def read_globe(self, path):
        globe = np.load(path)
        globe = globe[:,:,0:4]
        
        # print(globe.shape)
        # globe = globe.transpose(2,1,0)
        if globe.shape == (self.height, self.width, 4):
            pass
        else:
            globe = np.resize(globe, (self.height, self.width, 5))
            print_warning('Shape of the globe is not ({self.height}, {self.width}, 5)')
            print(str(path))
        return globe

    def __getitem__(self, index):
        # Read in Globe
        lidar = random.choice(self.lidars) # choice a lidar form lidars(LIST)
        weather_index = random.choice(range(self.weather_categories))
        weather = self.weathers[weather_index]
        
        globe_name = random.choice(self.npy_list[lidar][weather])
        
        GLOBE_PATH = Path(self.dataset_path).joinpath(lidar).joinpath(globe_name +'.npy')
        if GLOBE_PATH.is_file:
            globe = self.read_globe(GLOBE_PATH)
            # globe = Image.fromarray(globe)
            
        else:
            print_warning('NOT GET', str(GLOBE_PATH))
            return self.__getitem__(1)
        
        # To Tensor
        # if weather == 'light_fog_night':
            # print(weather)
        weather_index_in_WEATHERS = WEATHERS.index(weather)
        weather= torch.nn.functional.one_hot(torch.tensor(weather_index_in_WEATHERS), num_classes=len(WEATHERS)).type(torch.float32)

        globe = self.globe_transforms(globe)

        # TODO: labels for Object Detection
        if self.mode == 'train':
            return {'weather': weather, 'globe': globe}
        else: # Test or Validation
            return {'weather': weather, 'globe': globe, 'lidar': lidar, 'globe_name': globe_name, 'weather_name': self.weathers[weather_index], 'weather_index': weather_index}


    def __len__(self):
        
        if self.mode == 'valid':
            length = 0
            for lidar,weather in itertools.product(self.lidars, self.weathers):
                length = length + len(self.npy_list[lidar][weather])
        elif self.mode == 'train':
            return 32*5000
        else:
            return 10000
        return length

class SeeingThroughFogDataset2(Dataset):
    def __init__(self, 
                dataset_path='I:\Datasets\DENSE\SeeingThroughFog', 
                splits_path=r'.\data/Dense/SeeingThroughFog/splits', 
                mode = 'train',
                globe_height = 64,
                globe_width = 128,
                src_weather = 'clear_day',
                tgt_weathers = ['clear_day', 'clear_night' ,'dense_fog_day', 'dense_fog_night', 'light_fog_day', 'light_fog_night', 'rain', 'snow_day', 'snow_night'],
                lidars = ['lidar_hdl64_strongest']): #  'lidar_vlp32_strongest'
        # self.data_index = data_index
        self.mode = mode
        self.dataset_path = dataset_path
        self.src_weather = src_weather
        self.tgt_weathers = tgt_weathers

        self.lidars = lidars
        self.mode = mode
        self.height = globe_height
        self.width = globe_width


        # self.weather_categories = len(weathers)

        globe_transforms = [
                transforms.ToTensor(),
            ]
        if mode == 'train':
            globe_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
        else: # Test or Validation
            pass
        
        self.globe_transforms = transforms.Compose(globe_transforms)

        weathers = tgt_weathers[:]
        if src_weather not in weathers:
            weathers.append(src_weather)
        # print(tgt_weathers)
        self.npy_list = {} # self.bin_list[one of lidars][one of weathers] = one of names of globe in *.npy
        for lidar,weather in itertools.product(lidars, weathers):
            if lidar not in self.npy_list.keys():
                self.npy_list[lidar] = {}
            if weather == 'None':
                self.npy_list[lidar]['None'] = []
            else:
                with open(Path(splits_path).joinpath(mode).joinpath(weather+'.txt')) as f:
                    self.npy_list[lidar][weather] = [line.strip().replace(',', '_') for line in f.readlines()]

    def read_cloud(self, path):
        cloud = read_pcd(path)
        return cloud

    def read_globe(self, path):
        globe = np.load(path)
        # print(globe)
        globe = globe[:,:,0:4]
        # globe = globe.transpose(2,1,0)
        if globe.shape == (self.height, self.width,4):
            pass
        else:
            globe = np.resize(globe, (self.height, self.width, 5))
            print_warning(f'Shape of the globe is not ({self.height}, {self.width}, 5)')
            print(globe.shape)
            print(str(path))

        return globe

    def __getitem__(self, index):
        # Read in Globe
        lidar = random.choice(self.lidars) # choice a lidar form lidars(LIST)

        # scr_weather_index = random.choice(range(len(self.src_weathers)))
        # scr_weather = self.weathers[weather_index]
        
        src_globe_name = random.choice(self.npy_list[lidar][self.src_weather])
        
        GLOBE_PATH = Path(self.dataset_path).joinpath(lidar).joinpath(src_globe_name +'.npy')
        if GLOBE_PATH.is_file():
            # print(GLOBE_PATH)
            src_globe = self.read_globe(GLOBE_PATH)
            # globe = Image.fromarray(globe)
            
        else:
            print_warning('NOT GET' + str(GLOBE_PATH))
            return self.__getitem__(1)
        
        tgt_weather_index = random.choice(range(len(self.tgt_weathers)))
        tgt_weather = self.tgt_weathers[tgt_weather_index]
        tgt_globe_name = random.choice(self.npy_list[lidar][tgt_weather])

        GLOBE_PATH = Path(self.dataset_path).joinpath(lidar).joinpath(tgt_globe_name +'.npy')
        if GLOBE_PATH.is_file:
            tgt_globe = self.read_globe(GLOBE_PATH)
        else:
            print_warning('NOT GET', str(GLOBE_PATH))
            return self.__getitem__(1)

        # To Tensor
        # if weather == 'light_fog_night':
            # print(weather)
        weather_index_in_WEATHERS = WEATHERS.index(tgt_weather)
        weather= torch.nn.functional.one_hot(torch.tensor(weather_index_in_WEATHERS), num_classes=len(WEATHERS)).type(torch.float32)

        src_globe = self.globe_transforms(src_globe)
        tgt_globe = self.globe_transforms(tgt_globe)

        # TODO: labels for Object Detection
        if self.mode == 'train':
            return {'source': src_globe, 'target': tgt_globe, 'weather': weather}
        else: # Test or Validation
            return {'source': src_globe, 'target': tgt_globe, 'weather': weather, 'lidar': lidar, 
                    'src_globe_name': src_globe_name,  'tgt_globe_name': tgt_globe_name, 
                    'tgt_weather_name': self.tgt_weathers[tgt_weather_index], 'tgt_weather_index': tgt_weather_index}


    def __len__(self):
        if self.mode == 'train':
            return 32*5000
        elif self.mode == 'test':
            return 300
        else:
            return 1000


if __name__ == '__main__':
    import argparse
    from datetime import datetime
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader
    random.seed(datetime.now())
    torch.manual_seed(999)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="mu-log_var-test", help='name of the dataset')
    opt = parser.parse_args()


    dataset = ADUULMDataset(eval_mode=True)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True, collate_fn=collate_fn)
    cnt = 0
    for i, batch in enumerate(test_loader):
        print('here')

        weathers = ['sunny','foggy', 'night', 'rainy', 'snowy', 'snowyfoggyrainy']
        for weather in weathers:
            print('weather:', batch[weather].shape)

        cnt+=1
        if cnt >= 5:
            break

    # dataset = SeeingThroughFogDataset()
    # p = Path(r'/home/wanghejun/Desktop/wanghejun/WeatherShift/main/data/Dense/SeeingThroughFog/globe/lidar_hdl64_strongest/2018-02-03_20-48-35_00400.npy')
    # x = dataset.read_globe(path=p)
    # print(x.shape)
    # print(x.dtype)
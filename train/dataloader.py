
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import sys  
sys.path.append('/home/wanghejun/Desktop/wanghejun/WeatherShift/Weather-Shift')
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


class SeeingThroughFogDataset(Dataset): #TODO: Undo almost anything
    def __init__(self, 
                dataset_path='I:\Datasets\DENSE\SeeingThroughFog', 
                splits_path='.\splits', 
                mode = 'train',
                globe_height = 128,
                globe_width = 256,
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
                # transforms.Resize((globe_height, globe_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
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
        if path.is_file():
            image = cv2.imread(str(path))
            # globe = np.fromfile(path, dtype=np.float32)
            # globe = globe.reshape(self.height,self.width,3)
            globe = globe.transpose(2,1,0)
        else:
            print_warning('Not Found: '+ path)
            globe = None
        return globe

    def __getitem__(self, index):
        # Read in Globe
        lidar = random.choice(self.lidars) # choice a lidar form lidars(LIST)
        weather_index = random.choice(range(self.weather_categories))
        
        globe_name = random.choice(self.npy_list[lidar][self.weathers[weather_index]])
        
        GLOBE_PATH = Path(self.dataset_path).joinpath(lidar).joinpath(globe_name +'.npy')
        if GLOBE_PATH.is_file:
            globe = self.read_globe(GLOBE_PATH)
            globe = Image.fromarray(globe)
            
        else:
            print_warning('NOT GET', str(GLOBE_PATH))
            return self.__getitem__(1)

        # To Tensor
        weather= torch.nn.functional.one_hot(torch.tensor(weather_index), num_classes=self.weather_categories+1).type(torch.float32)

        print('1',globe.size)

        globe = self.globe_transforms(globe)
        print('2',globe.size())


        # TODO: labels for Object Detection
        if self.mode == 'train':
            return {'weather': weather, 'globe': globe}
        else: # Test or Validation
            return {'weather': weather, 'globe': globe, 'lidar': lidar, 'globe_name': globe_name, 'weather_name': self.weathers[weather_index], 'weather_index': weather_index}


    def __len__(self):
        length = 0
        for lidar,weather in itertools.product(self.lidars, self.weathers):
            length = length + len(self.npy_list[lidar][weather])
        return length


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
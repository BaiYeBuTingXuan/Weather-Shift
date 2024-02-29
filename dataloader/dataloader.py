
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

from utils import get_filelist,read_pcd,print_warning
from utils.math import angle_normal,spin,sign


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


class ADUULMDataset(Dataset):
    def __init__(self, dataset_path='/home/wanghejun/Desktop/wanghejun/WeatherShift/Weather-Shift/data/ADUULM', 
                granularity=1000, eval_mode=False, n_points=1024,
                weathers = ['sunny','foggy', 'night', 'rainy', 'snowy', 'snowyfoggyrainy']):
        # self.data_index = data_index
        self.eval_mode = eval_mode
        self.dataset_path = dataset_path
        self.granularity = granularity
        
        pointcloud_transforms = [
            # transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
            # transforms.Normalize((0.5), (0.5))
        ]
        
        self.pointcloud_transforms = transforms.Compose(pointcloud_transforms)
        self.weathers = weathers
        self.npy_list = {}
        for weather in self.weathers:
            self.npy_list[weather] = get_filelist(os.path.join(self.dataset_path, 'sorted', weather, '*', 'syn', '*.npy'), only_name=False)


    
    def read_npy(self, path2npy):
        if os.path.isfile(path2npy):
            data = np.load(path2npy)
            print(data)
            print(path2npy)

            return data
        else:
            print('here')
            print_warning('NOT Found '+path2npy)
            return None
    
    def read_cloud(self, weather, lidar, kind, name):
        path2npy = os.path.join(self.dataset_path, 'sorted', weather, name, kind, lidar+'.pcd')
        data = self.read_npy(path2npy)
        return data

    def random_domainness(self, index):
        return torch.randint(0, self.granularity, (1))

    def __getitem__(self, index):
        # Read in Cloud
        # print('1')
        cloud = {}
        domanness = random.randint(0, self.granularity)
        for weather in self.weathers:
            path2npy = random.choice(self.npy_list[weather])
            cloud[weather] = self.read_npy(path2npy) # without RGB, point = [x y z i]
            print(cloud[weather].shape)
            # print('2'+weather)
            
            # mirror the inputs
            for d in [0,1,2,3]:
                mirror = True if random.uniform(0.0, 1.0) > 0.5 else False
                if mirror:
                    cloud[weather][d] = cloud[weather][d]*(-1.0)
            
            print(cloud[weather].shape)

            cloud[weather] = self.pointcloud_transforms(cloud[weather])

        if not self.eval_mode:
            result =  cloud
        else:
            result =  cloud

        return result

    def __len__(self):
        length = [len(self.npy_list[weather]) for weather in self.weathers]
        return sum(length)


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
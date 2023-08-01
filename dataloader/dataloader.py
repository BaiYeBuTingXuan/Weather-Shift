
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
from os import path
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
    
def angle_normal(angle):
    while angle >= np.pi:
        angle -= 2*np.pi
    while angle <= -np.pi:
        angle += 2*np.pi
    return angle
    
def get_filelist(path, index:int):
    files = glob.glob(path+'/'+str(index)+'/ipm/*.png')
    file_names = []
    for file in files:
        file_name = file.split('/')[-1][:-4]
        # check for img pic validity
        # if cv2.imread(path+'/'+str(index)+'/img/'+file_name+'.png') is None:
        #     print('not found:',file_name)
        #     continue
        # # check for nav pic validity
        # pic = cv2.imread(path+'/'+str(index)+'/nav/'+file_name+'.png')
        # if not check_nav_valid(pic):
        #     continue

        file_names.append(file_name)
    file_names.sort()
    return file_names

def spin(xy,deg):
    x,y = xy
    rad = deg/180*np.pi

    # rotation equation:
    # [x] = [cost -sint][x']
    # [y] = [sint  cost][y']
    
    x_ = x*np.cos(rad)-y*np.sin(rad)
    y_ = x*np.sin(rad)+y*np.cos(rad)
    # res = ()
    return [x_,y_]

def sign(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    else:
        return 0


class CARLADataset(Dataset):
    def __init__(self, data_index, dataset_path, granularity=1000, eval_mode=False, n_points=1024):
        self.data_index = data_index
        self.eval_mode = eval_mode
        self.dataset_path = dataset_path
        self.granularity = granularity
        
        pointcloud_transforms = [
            # transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
        
        self.pointcloud_transforms = transforms.Compose(pointcloud_transforms)
    
    def read_cloud(self, index):
        pass

    def random_domainness(self, index):
        return torch.randint(0, self.granularity, (1))

    def __getitem__(self, index):
        # Read in Cloud
        cloud = None
        domanness = random.randint(0, self.granularity)

        # Transfer Domainness
        
        # mirror the inputs
        mirror = True if random.uniform(0.0, 1.0) > 0.5 else False
        if mirror:
            pass
            #     break

            # except:
            #     pass

        
        cloud = self.img_transforms(cloud)        
            
        # if not self.eval_mode:
        #     return {'img_nav': input_img, 'label': label, 'fake_nav_with_img':fake_input_img, 'seg_nav':input_seg}
        # else:
        #     return {'img': img, 'nav': nav, 'fake_nav':fake_nav, 'label': label, 'file_name':file_name,'seg':seg}

    def __len__(self):
        return 160000


if __name__ == '__main__':
    import argparse
    from datetime import datetime
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader
    random.seed(datetime.now())
    torch.manual_seed(999)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="mu-log_var-test", help='name of the dataset')
    parser.add_argument('--width', type=int, default=400, help='image width')
    parser.add_argument('--height', type=int, default=200, help='image height')
    parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
    parser.add_argument('--traj_steps', type=int, default=8, help='traj steps')
    parser.add_argument('--max_dist', type=float, default=25., help='max distance')
    parser.add_argument('--max_t', type=float, default=3., help='max time')
    parser.add_argument('--beta1', type=float, default=1, help='beta parameter')
    parser.add_argument('--beta2', type=float, default=1, help='beta parameter')
    opt = parser.parse_args()

    test_loader = DataLoader(CARLADataset([21], dataset_path='../datacollect/DATASET/CARLA/Segmentation/', eval_mode=True),
                         batch_size=1, shuffle=False, num_workers=1, pin_memory=True,persistent_workers=True)

    cnt = 0
    for i, batch in enumerate(test_loader):
        # return {'img': img, 'nav': nav, 'fake_nav':fake_nav, 'label': label, 'file_name':file_name,'seg':seg}
        img = batch['img'].clone().data.numpy().squeeze()*127+128
        nav = batch['nav'].clone().data.numpy().squeeze()*127+128
        
        img = np.transpose(img,(1,2,0))
        nav = np.transpose(nav,(1,2,0))

        # print(img.shape)
        print('file_name',batch['file_name'])
        # img = Image.fromarray(np.transpose(img, (2, 1, 0)).astype('uint8')).convert("RGB")
        # nav = Image.fromarray(np.transpose(nav, (2, 1, 0)).astype('uint8')).convert("RGB")

        cv2.imwrite('./img_%d.jpg'%cnt,img)
        cv2.imwrite('./nav_%d.jpg'%cnt,nav)



        cnt+=1
        if cnt >= 5:
            break
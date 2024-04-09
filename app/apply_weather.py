import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import argparse
import random

import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from train.dataloader import SeeingThroughFogDataset, WEATHERS
from utils.point_cloud import read_pcd,nor_globe_voxelization,anti_globe_voxelization,Param
from models.unet import UNetGenerator_Normal as Generator

random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)

parser = argparse.ArgumentParser()
parser.add_argument('--lidars', type=list, default=['lidar_hdl64_strongest'], help='list of train lidar')
parser.add_argument('--model_name', type=str, default="Simulator/G", help='path to the model')
parser.add_argument('--weathers', type=int, default=['dense_fog_day'], help='saved epoch of the model for test')

opt = parser.parse_args()

BASE_PATH = Path('/home/wanghejun/Desktop/wanghejun/WeatherShift/main/')
PATH_TO_MODEL_G = BASE_PATH.joinpath('models/trained').joinpath(opt.model_name+'.pth')
PATH_TO_DATA = BASE_PATH.joinpath('data/Dense/SeeingThroughFog')
PATH_TO_PARAM = Path('./utils/lidar_param.json')

PATH_TO_CLEAR_WEATHER_SPLITS =BASE_PATH.joinpath('data/Dense/SeeingThroughFog/splits/origin').joinpath('clear_day.txt')
globe_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:',device)

G = Generator().to(device)
print('Generator:', PATH_TO_MODEL_G)
G.load_state_dict(torch.load(PATH_TO_MODEL_G))
G.eval()

if __name__ == '__main__':
    for lidar,weather in zip(opt.lidars, opt.weathers):
        print('lidar:',lidar)
        print('target weather:',weather)


        PATH_TO_ORIGIN_DATA = PATH_TO_DATA.joinpath('cloud').joinpath(lidar)
        PATH_TO_FAKE_DATA = PATH_TO_DATA.joinpath(weather).joinpath(lidar)
        PATH_TO_FAKE_DATA.mkdir(parents=True, exist_ok=True)

        param = Param(lidar, PATH_TO_PARAM)

        with open(PATH_TO_CLEAR_WEATHER_SPLITS,'r') as file:
            filenames = [l.strip().replace(',','_')+'.bin' for l in file.readlines ()]
        bar = enumerate(filenames)
        length = len(filenames)
        bar = tqdm(bar, total=length)

        for _, origin_cloud in bar:
            orgin_pc = read_pcd(PATH_TO_ORIGIN_DATA.joinpath(origin_cloud))
            globe, _ = nor_globe_voxelization(orgin_pc,param)
            input_globe = globe_transforms(globe[:,:,0:5]).to(device)

            weather_index_in_WEATHERS = WEATHERS.index(weather)
            target_weather= torch.nn.functional.one_hot(torch.tensor(weather_index_in_WEATHERS), num_classes=len(WEATHERS)).type(torch.float32).to(device)

            input_globe = input_globe.unsqueeze(0)
            target_weather = target_weather.unsqueeze(0)

            _, (alpha,rho,radius,reflectance) = G(input_globe,target_weather)
            alpha = alpha[0].detach().cpu().numpy()
            rho = rho[0].detach().cpu().numpy()
            radius = radius[0].detach().cpu().numpy()
            reflectance = reflectance[0].detach().cpu().numpy()

            modified_args = (alpha,rho,radius,reflectance)
            
            fake_pc = anti_globe_voxelization(globe, modified_args, param)
            fake_pc.tofile(PATH_TO_FAKE_DATA.joinpath(origin_cloud+'.bin'))
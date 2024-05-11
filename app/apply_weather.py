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
import yaml

import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from train.dataloader import SeeingThroughFogDataset, WEATHERS
from utils.point_cloud import read_pcd,nor_globe_voxelization,anti_globe_voxelization,Param,quick_anti_globe_voxelization
from models.unet import UNetGenerator_Normal as Generator

from LiDAR_fog_sim.fog_simulation import simulate_fog, ParameterSet

from itertools import product

random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)

parser = argparse.ArgumentParser()
parser.add_argument('--lidars', type=list, default=['lidar_hdl64_strongest'], help='list of train lidar')
parser.add_argument('--model_name', type=str, default="Simulator/model_400000", help='path to the model')
parser.add_argument('--weathers', type=list, default=['dense_fog_night', 'rain', 'snow_day', 'snow_night'], help='saved epoch of the model for test')
parser.add_argument('--method', type=str, default='generated',help='saved epoch of the model for test')

opt = parser.parse_args()
['clear_night','light_fog_day','dense_fog_night']
['dense_fog_day', 'light_fog_night', 'rain', 'snow_day', 'snow_night']
['clear_night','light_fog_day','dense_fog_day', 'light_fog_night', 'dense_fog_night', 'rain', 'snow_day', 'snow_night']
BASE_PATH = Path('/home/wanghejun/Desktop/wanghejun/WeatherShift/main/')
PATH_TO_MODEL_G = BASE_PATH.joinpath('models/trained').joinpath(opt.model_name+'.pth')
PATH_TO_DATA = BASE_PATH.joinpath('data/Dense/SeeingThroughFog')
PATH_TO_PARAM = Path('./utils/lidar_param.json')

PATH_TO_CLEAR_WEATHER_SPLITS =BASE_PATH.joinpath('data/Dense/SeeingThroughFog/splits/origin').joinpath('clear_day.txt')
globe_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

if opt.method == 'generated' or 'generated2':
    print('Apply weather using generative model')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:',device)
    G = Generator(in_channels=4).to(device)
    print('Generator:', PATH_TO_MODEL_G)
    G.load_state_dict(torch.load(PATH_TO_MODEL_G))
    G.eval()
else:
    print('Apply weather using physical simulator')
    alpha_list = [0, 0.005, 0.01, 0.02, 0.03, 0.06]
    print('Alpha uniformly sample from:', alpha_list)
    

def generate(orgin_pc,weather,param):
    # print(orgin_pc[0])
    globe, _ = nor_globe_voxelization(orgin_pc,param)
    #print('here the origin pc has been to Spher from cart')
    # print(orgin_pc[0])
    input_globe = globe_transforms(globe[:,:,0:4]).to(device)

    weather_index_in_WEATHERS = WEATHERS.index(weather)
    target_weather= torch.nn.functional.one_hot(torch.tensor(weather_index_in_WEATHERS), num_classes=len(WEATHERS)).type(torch.float32).to(device)

    input_globe = input_globe.unsqueeze(0)
    target_weather = target_weather.unsqueeze(0)

    _, (alpha,radius,reflectance) = G(input_globe,target_weather)
    alpha = alpha[0].detach().cpu().numpy()
    # rho = rho[0].detach().cpu().numpy()
    radius = radius[0].detach().cpu().numpy()
    reflectance = reflectance[0].detach().cpu().numpy()

    modified_args = (alpha,radius,reflectance)
    # print(orgin_pc[0])
    # exit()
    fake_pc,number_change = quick_anti_globe_voxelization(orgin_pc, modified_args, param)
    # fake_pc.tofile(PATH_TO_FAKE_DATA.joinpath(origin_cloud))

    return fake_pc,number_change


def generate2(orgin_pc,weather,param):
    # print(orgin_pc[0])
    globe, _ = nor_globe_voxelization(orgin_pc,param)
    #print('here the origin pc has been to Spher from cart')
    # print(orgin_pc[0])
    input_globe = globe_transforms(globe[:,:,0:4]).to(device)

    weather_index_in_WEATHERS = WEATHERS.index(weather)
    target_weather= torch.nn.functional.one_hot(torch.tensor(weather_index_in_WEATHERS), num_classes=len(WEATHERS)).type(torch.float32).to(device)

    input_globe = input_globe.unsqueeze(0)
    target_weather = target_weather.unsqueeze(0)

    _, (alpha,radius,reflectance) = G(input_globe,target_weather)
    alpha = alpha[0].detach().cpu().numpy()
    # rho = rho[0].detach().cpu().numpy()
    radius = radius[0].detach().cpu().numpy()
    reflectance = reflectance[0].detach().cpu().numpy()

    modified_args = (alpha,radius,reflectance)
    # print(orgin_pc[0])
    # exit()
    fake_pc,number_change = anti_globe_voxelization(globe, modified_args, param)
    # fake_pc.tofile(PATH_TO_FAKE_DATA.joinpath(origin_cloud))

    return fake_pc,number_change

def simulate(orgin_pc,weather,param):
    CONGFIG = './physical_simulation.yaml'
    try:
        config = yaml.safe_load(CONGFIG, Loader=yaml.FullLoader)
    except:
        config = yaml.safe_load(CONGFIG)

    if weather == 'dense_fog_day':
        # def simulate_fog(p: ParameterSet, pc: np.ndarray, noise: int, gain: bool = False, noise_variant: str = 'v1',
                #  hard: bool = True, soft: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        alpha = random.choice(alpha_list)
        p = ParameterSet(alpha=alpha)
        # print('here')
        noise_variant = 'v4'
        noise = 20
        pc, fake_pc, _ = simulate_fog(p, orgin_pc, noise = 20, noise_variant = 'v4')
        return pc,len(fake_pc)




if __name__ == '__main__':

    if opt.method == 'generated':
        func = generate
    # if opt.method == 'generated2':
    #     func = generate2
    else:
        func = simulate
        pass

    for lidar,weather in product(opt.lidars, opt.weathers):
        print('lidar:',lidar)
        print('target weather:',weather)


        PATH_TO_ORIGIN_DATA = PATH_TO_DATA.joinpath('cloud').joinpath(lidar)
        PATH_TO_FAKE_DATA = PATH_TO_DATA.joinpath(opt.method).joinpath(weather).joinpath(lidar)
        PATH_TO_FAKE_DATA.mkdir(parents=True, exist_ok=True)

        param = Param(lidar, PATH_TO_PARAM)

        with open(PATH_TO_CLEAR_WEATHER_SPLITS,'r') as file:
            filenames = [l.strip().replace(',','_')+'.bin' for l in file.readlines ()]
        bar = enumerate(filenames)
        length = len(filenames)
        bar = tqdm(bar, total=length)

        for _, origin_cloud in bar:
           orgin_pc = read_pcd(PATH_TO_ORIGIN_DATA.joinpath(origin_cloud))

           fake_pc,change_num = func(orgin_pc,weather,param)
           fake_pc = fake_pc.astype(np.float32)
        #    print(fake_pc.dtype)
           fake_pc.tofile(PATH_TO_FAKE_DATA.joinpath(origin_cloud))
           bar.set_description('origin %d to fake %d (%d changed)' % (len(orgin_pc),len(fake_pc),change_num))
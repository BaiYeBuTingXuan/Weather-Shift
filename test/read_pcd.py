import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

from utils.point_cloud import read_pcd
from pathlib import Path
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    weather = 'dense_fog_day'
    lidar = 'lidar_hdl64_strongest'

    BASE_PATH = Path('/home/wanghejun/Desktop/wanghejun/WeatherShift/main/')
    PATH_TO_DATA = BASE_PATH.joinpath('data/Dense/SeeingThroughFog')
    PATH_TO_REAL_DATA = PATH_TO_DATA.joinpath('cloud').joinpath(lidar)
    PATH_TO_FAKE_DATA = PATH_TO_DATA.joinpath('generated').joinpath(weather).joinpath(lidar)
    PATH_TO_CLEAR_WEATHER_SPLITS =BASE_PATH.joinpath('data/Dense/SeeingThroughFog/splits/origin').joinpath('clear_day.txt')

    with open(PATH_TO_CLEAR_WEATHER_SPLITS,'r') as file:
        filenames = [l.strip().replace(',','_')+'.bin' for l in file.readlines ()]
        bar = enumerate(filenames)
        length = len(filenames)
        bar = tqdm(bar, total=length)

    for i, filename in bar:
        # print(PATH_TO_FAKE_DATA)
        tensor = read_pcd(PATH_TO_FAKE_DATA.joinpath(filename))
        has_inf = np.isinf(tensor).any()
        has_nan = np.isnan(tensor).any()
        if has_inf or has_nan:
            print('====================')
            print(filename)

        max_values_per_dimension= np.max(tensor,axis=0)
        min_values_per_dimension= np.min(tensor,axis=0)
        
        if i == 0:
            pass
        else:
            max_values_per_dimension = np.stack([max_values_per_dimension,np.max(tensor,axis=0)],axis=0)
            min_values_per_dimension = np.stack([min_values_per_dimension,np.min(tensor,axis=0)],axis=0)

    # max_values_per_dimension = np.max(max_values_per_dimension,axis=0)
    # min_values_per_dimension = np.min(min_values_per_dimension,axis=0)
    max_values_per_dimension = np.percentile(max_values_per_dimension, q=75, axis=0)
    min_values_per_dimension = np.percentile(min_values_per_dimension, q=25, axis=0)

    print("Max values per dimension:", max_values_per_dimension)
    print("Min values per dimension:", min_values_per_dimension)


# TODO : transfer *.bin of points cloud to *.png in globe
import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
import numpy as np

from pathlib import Path

from utils import print_warning,ndarray2img
from utils.math_ import Cart2Cylin,approx_equal,Cart2Spher,PI
from utils.point_cloud import *

import cv2
from tqdm import tqdm
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lidar_name', type=str, default="lidar_hdl64_last", help='name of the lidar') # lidar_hdl64_strongest lidar_vlp32_strongest
args = parser.parse_args()

STF_PATH = Path('/home/wanghejun/Desktop/wanghejun/WeatherShift/main/data/Dense/SeeingThroughFog')
PATH_TO_PARAM = Path('./utils/lidar_param.json')


# WEIGHT,HEIGHT = 256, 128

if __name__ == '__main__':
    SRC_PATH = STF_PATH.joinpath('cloud').joinpath(args.lidar_name)
    SAVE_PATH = STF_PATH.joinpath('globe').joinpath(args.lidar_name)

    SAVE_PATH.mkdir(parents=True, exist_ok = True)
    if STF_PATH.is_dir():
        files = list(SRC_PATH.glob('*.bin'))
        files.sort(key=str)
    else:
        print_warning('Not Found '+str(SRC_PATH))
        sys.exit()

    param = Param(args.lidar_name, PATH_TO_PARAM)
    print(param)
    # print('lidar name:', args.lidar_name)
    bar = enumerate(files)
    bar = tqdm(bar, desc="Processing", total=len(files))
    try:
        for _, file in bar:
            bar.desc = str(file.stem)
            pc = read_pcd(file)
            frame,_ = nor_globe_voxelization(pc, param=param)
            # print(frame.shape)
            # frame = ndarray2img(frame)    
            np.save(str(SAVE_PATH.joinpath(file.stem+'.npy')), frame)
            # print(SAVE_PATH.joinpath(file.stem+'.npy'))
            # cv2.imwrite(str(SAVE_PATH.joinpath(file.stem+'.bin')), frame)
            # cv2.waitKey(0)
    except KeyboardInterrupt:
        pass

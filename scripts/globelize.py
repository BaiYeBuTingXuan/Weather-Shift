# TODO : transfer *.bin of points cloud to *.png in globe
import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
import numpy as np

from pathlib import Path

from utils import print_warning,ndarray2img
from utils.math import Cart2Cylin,approx_equal,Cart2Spher,PI
from utils.point_cloud import *

import cv2
from tqdm import tqdm
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lidar_name', type=str, default="lidar_hdl64_strongest", help='name of the lidar')
args = parser.parse_args()

STF_PATH = Path('I:\Datasets\DENSE\SeeingThroughFog')
SAVE_PATH = Path('.\datasets\DENSE\SeeingThroughFog').joinpath(args.lidar_name)

WEIGHT,HEIGHT = 256, 128

if __name__ == '__main__':
    SAVE_PATH.mkdir(parents=True, exist_ok = True)

    if STF_PATH.is_dir():
        files = list(STF_PATH.joinpath('args.lidar_name').glob('*.bin'))
        files.sort(key=str)
    else:
        print_warning('Not Found '+str(STF_PATH))
        sys.exit()

    bar = enumerate(files)
    bar = tqdm(bar, desc="Processing", total=len(files))
    try:
        for _, file in bar:
            bar.desc = str(file.stem)
            pc = read_pcd(file)
            frame = globe_voxelization(pc, HEIGHT, WEIGHT)
            # frame = ndarray2img(frame)
            np.save(SAVE_PATH.joinpath(file.stem+'.npy'), frame)
            # cv2.waitKey(0)
    except KeyboardInterrupt:
        pass

import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
import numpy as np

from pathlib import Path

from utils import print_warning,ndarray2img

import argparse
import json
import numpy as np
import cv2
from datetime import datetime
from tqdm import tqdm
from SeeingThroughFog.tools.DatasetViewer.lib.read import load_calib_data, get_kitti_object_list

parser = argparse.ArgumentParser()
parser.add_argument('--lidar_name', type=str, default="lidar_hdl64_strongest", help='name of the lidar')
args = parser.parse_args()

CUR_PATH = Path('/home/wanghejun/Desktop/wanghejun/WeatherShift/main')
STF_PATH = CUR_PATH.joinpath('data/Dense/SeeingThroughFog')
CALIB_PATH = CUR_PATH.joinpath('SeeingThroughFog/tools/DatasetViewer/calibs')

SRC_PATH = STF_PATH.joinpath('gt_labels/cam_left_labels_TMP')
TGT_PATH = STF_PATH.joinpath('gt_labels/lidar3D')

TGT_PATH.mkdir(parents=True, exist_ok=True)

root = str(CALIB_PATH)
tf_tree = 'calib_tf_tree_full.json'
name_camera_calib = 'calib_cam_stereo_left.json'

_, camera_to_velodyne, P, _, _, _, zero_to_camera = load_calib_data(root, name_camera_calib, tf_tree)

if SRC_PATH.is_dir():
    files = list(SRC_PATH.glob('*.txt'))
    files.sort(key=str)
else:
    print_warning('Not Found '+str(SRC_PATH))
    sys.exit()

bar = enumerate(files)
bar = tqdm(bar, desc="Processing", total=len(files))
all_class = []

try:
    for _, file in bar:
        bar.desc = str(file.stem)
        objects = get_kitti_object_list(str(file), camera_to_velodyne=camera_to_velodyne)

        tgt_file = TGT_PATH.joinpath(file.name)
        with open(tgt_file,'w') as f:
            for obj in objects:
                cx,cy,cz = obj['posx_lidar'], obj['posy_lidar'], obj['posz_lidar']
                dx,dy,dz = obj['length'], obj['width'], obj['height']
                heading = -1*obj['rotz']
                name = obj['identity']
                combined_str = f"{name} {cx} {cy} {cz} {dx} {dy} {dz} {heading}\n"
                f.write(combined_str)
            if name not in all_class or len(all_class) == 0:
                all_class.append(name)
except KeyboardInterrupt:
    pass

print('All Class:')
for c in all_class:
    print('   -',c)

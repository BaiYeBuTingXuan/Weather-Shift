import numpy as np
import argparse
import os
import sys  
sys.path.append('/home/wanghejun/Desktop/wanghejun/WeatherShift/Weather-Shift')
from glob import glob
from utils import mkdir, get_filelist, read_pcd
from tqdm import tqdm
import re

parser = argparse.ArgumentParser(description='sort ADUULM')
# DEFAULT_SRC_PATH = '/home/wanghejun/Desktop/wanghejun/WeatherShift/Weather-Shift/data/ADUULM/dataset/data/sunny/Lehr_sunny_a_000000005_1538662846738852856/Lehr_sunny_a_000000005_1538662846738852856_VeloFront_syn.pcd'
DEFAULT_SRC_PATH = '/home/wanghejun/Desktop/wanghejun/WeatherShift/Weather-Shift/data/ADUULM/dataset'
DEFAULT_TGT_PATH = '/home/wanghejun/Desktop/wanghejun/WeatherShift/Weather-Shift/data/ADUULM/sorted'


# 添加命令行参数和选项
parser.add_argument('--src_path', type=str, default=DEFAULT_SRC_PATH, help='path to the rare PointCloud file')
parser.add_argument('--tgt_path', type=str, default=DEFAULT_TGT_PATH, help='path to the sorted PointCloud file')
parser.add_argument('--lidar_name', type=str, default='VeloFront', help='VeloFront,VeloLeft,VeloRight or VeloRear')

# 解析命令行参数
args = parser.parse_args()

def sort(weathers):
    for weather in weathers:
        SRC_PATH = os.path.join(args.src_path, 'data', weather)
        TGT_PATH = os.path.join(args.tgt_path, weather)

        # glob read file name list
        filelist = get_filelist(os.path.join(SRC_PATH, '*', '*.pcd'))
        filelist = [f for f in filelist if not re.search(r'_labeled$', f)]
        # pcd --> npy
        for file in tqdm(filelist, desc=weather):
            name = file[:-4]
            name_spilt = name.rsplit('_',2)
            father, lidar, kind = name_spilt[0],name_spilt[1],name_spilt[2]
            # father : father fold of the files
            # lidar: lidar position like VeloFront,VeloLeft,VeloRight or VeloRear
            # kind : syn or labeled

            pcd_path = os.path.join(SRC_PATH, father, file)
            pcd_arr = read_pcd(pcd_path)
            
            tgt_path = os.path.join(TGT_PATH, father, kind)
            mkdir(tgt_path)

            tgt_path = os.path.join(tgt_path, lidar+'.npy')
            np.save(tgt_path, pcd_arr)
    
if __name__ == '__main__':
    weathers = ['sunny','foggy', 'night', 'rainy', 'snowy', 'snowyfoggyrainy']
    sort(weathers)
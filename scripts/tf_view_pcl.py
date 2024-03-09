from tkinter import VERTICAL
import numpy as np
import argparse
import os
import sys  
# import open3d as o3d
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append('/home/wanghejun/Desktop/wanghejun/WeatherShift/Weather-Shift')

from utils.point_cloud import read_pcd

parser = argparse.ArgumentParser(description='这是一个示例程序')

BASE_PATH = '/home/wanghejun/Desktop/wanghejun/WeatherShift/Weather-Shift'
DEFAULT_PATH = BASE_PATH+'/data/ADUULM/dataset/data/sunny//UlmOchensteige_sunny_a_000000013_1538664092334990214/UlmOchensteige_sunny_a_000000013_1538664092334990214_VeloFront_labeled.pcd'
# /home/wanghejun/Desktop/wanghejun/WeatherShift/Weather-Shift/data/ADUULM/dataset/data/sunny/UlmOchensteige_sunny_a_000000013_1538664092334990214/UlmOchensteige_sunny_a_000000013_1538664092334990214_VeloFront_labeled.pcd
LOG_PATH = BASE_PATH + '/scripts/'

# 添加命令行参数和选项
parser.add_argument('--path', default=DEFAULT_PATH, help='path to the PointCloud file')
parser.add_argument('--color', default=[255,255,0], help='[R,G,B] for all points')


# 解析命令行参数
args = parser.parse_args()

if __name__ == '__main__':
    # print("load the points ...")
    # points = read_pcd(args.path)

    # print("construct the pcd ...")
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)

    # print("create the window ...")
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # vis.add_geometry(pcd)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.run()

    # cv2.waitKey(0)
    # # print(pcd_array[0:10,:])
    # # print(pcd_array.shape)
    
    points = read_pcd(args.path)

    points = torch.tensor(np.asarray(points)) # 转换为tensor

    vertices = points

    colors = torch.zeros_like(vertices)
    # faces = torch.zeros_like(vertices)


    for i in [0,1,2]:
        colors[:,i] = args.color[i]/255.0

    writer = SummaryWriter(log_dir=LOG_PATH)
    
    # visualize the point cloud
    writer.add_mesh('example', vertices=vertices.unsqueeze(0), colors=colors.unsqueeze(0))
    # print('here')

    writer.close()
        
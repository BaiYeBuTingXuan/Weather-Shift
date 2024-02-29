import rosbag as bag
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
sys.path.append("/home/wanghejun/Desktop/wanghejun/WeatherShift/Weather-Shift/")
import numpy as np
import argparse
import time
import cv2
from tqdm import tqdm



bag_path = "/home/wanghejun/Desktop/wanghejun/WeatherShift/Weather-Shift/data/ADUULM/ADUULM_Dataset/demo_sequences/sunny.bag"
data_path = "/home/wanghejun/Desktop/wanghejun/WeatherShift/Weather-Shift/data/ADUULM/dataset/"

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-T', '--topic', type=str, default='/lidar/velodyne_fc/velodyne_points', help='topic to decode')
# parser.add_argument('-n', '--num', type=int, default=100000, help='total number')
args = parser.parse_args()

def mkdir(path):
    if not os.path.exists(data_path+path):
        os.makedirs(data_path+path)

# mkdir('lidar/velodyne_fc/velodyne_points/') # Velodyne前置屋顶激光雷达点云数据
# mkdir('lidar/velodyne_fl/velodyne_points/') # Velodyne前左侧激光雷达数据包
# mkdir('lidar/velodyne_fr/velodyne_points/') # Velodyne前右侧激光雷达点云数据
# mkdir('lidar/velodyne_rc/velodyne_points/:') # Velodyne后屋顶激光雷达点云数据


topic_name = ['/lidar/velodyne_fc/velodyne_points/','/lidar/velodyne_fl/velodyne_points/','/lidar/velodyne_fr/velodyne_points/','/lidar/velodyne_rc/velodyne_points/']


# 打开ROS包文件
bag = bag.Bag(bag_path)
for name in topic_name:
# 遍历ROS包中指定主题的消息
  mkdir(name)
  for topic, msg, t in tqdm(bag.read_messages(topics=[name]),desc=name):
    # 创建文件名，使用时间戳作为文件名
    file_name = data_path + name + str(t.to_nsec()) + '.txt'

    # 将指定主题的消息保存到文件
    with open(file_name, 'w') as file:
      file.write(str(msg))

# 关闭ROS包文件
bag.close()


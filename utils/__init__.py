import numpy as np
import argparse
import os
import sys  
sys.path.append('/home/wanghejun/Desktop/wanghejun/WeatherShift/Weather-Shift')
import glob

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_filelist(path,only_name=True):
    files = glob.glob(path)
    # print(files)
    file_names = []
    for file in files:
        if only_name:
            file_name = os.path.split(file)[-1]
        else:
            file_name = file
        file_names.append(file_name)
    file_names.sort()
    return file_names

def read_pcd(pcd_file):
    # pcd --> numpy.ndarrray
    # [...,...,[x,y,z,i,rgb],...,...] dtype = float
    line_number = 0
    pointcloud = []
    # open file
    if (os.path.isfile(pcd_file)):
        image_file = open(pcd_file,"r")
        for line in image_file:
            line_number = line_number + 1
            if line_number > 11: # Point: x,y,z,rgb=0,i
                try:
                    buff = line.split()

                    x = float(buff[0])
                    y = float(buff[1])
                    z = float(buff[2])
                    # rgb = float(buff[3])
                    # print(dec2rgb(rgb))
                    # i = float(buff[4])
                    
                    point = [x,y,z] 
                    pointcloud.append(point)
                except IndexError:
                    print('IndexError')
                    print(pcd_file)


    pointcloud = np.array(pointcloud)

    return pointcloud # as ndarray n*5 [x,y,z,i,rgb]

def print_warning(message):
    print('\033[93m [WARNING]' + message + '\033[0m')



# def dec2rgb(dec):
#   x = int(dec)

#   print(hex(x))
#   print(x)
#   print(dec)

#   r = (x & 0xFF << 24) >> 24
#   g = (x & 0xFF << 16) >> 16
#   b = (x & 0xFF << 8) >> 8
#   _ = x & 0xFF
#   return [r, g, b, _]
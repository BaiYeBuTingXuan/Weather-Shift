import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
import numpy as np

from pathlib import Path

from utils import print_warning,ndarray2img
from utils.math import Cart2Cylin,approx_equal,Cart2Spher,PI

import cv2
from tqdm import tqdm
import itertools
import torch
import json

class Param:
    def __init__(self, lidar, path2json) -> None:
    
        with open(path2json, 'r') as json_file:
            args = json.load(json_file)[lidar]
            # print(args)
            for key, value in args.items():
                if key == 'latitude_n' or key == 'longitude_n':
                    setattr(self, key, value)
                elif key == 'latitude_bound' or key == 'longitude_bound':
                    value = [eval(v) for v in value]
                    setattr(self, key, value)


        self.min_bound = np.array([self.latitude_bound[0], self.longitude_bound[0]], dtype=np.float32)
        self.max_bound = np.array([self.latitude_bound[1], self.longitude_bound[1]], dtype=np.float32)

        self.resolution = (self.max_bound - self.min_bound)/np.array([self.latitude_n, self.longitude_n], dtype=float)
        


def globe_voxelization(pc:np.array, param:Param) ->np.array:
    '''
    input: points cloud n*[x,y,z,i]

    output: latitude_n * longitude_n * 3
        per pixel : [number of points, average radius of points, average reflected reflectance of points]

    latitude : Angle between Point with Z-Positive [-PI/2, PI/2]
    longitude : Angle on XY-Plane (-PI, PI]

    '''
    pc[:,0:3] = Cart2Spher(x=pc[:,0],y=pc[:,1],z=pc[:,2])
    # angles = pc[:, 0:2]

    voxels = np.zeros((param.latitude_n, param.longitude_n, 3), dtype=float)

    lats = []


    for p in pc:
        latitude = p[0]
        longitude = p[1]
        radius = p[2]
        reflectance = p[3]
        # print(p)
        i = int((latitude-param.min_bound[0]) / param.resolution[0])
        j = int((longitude-param.min_bound[1]) / param.resolution[1])

        i = min(param.latitude_n-1, i)
        j = min(param.longitude_n-1, j)

        voxels[i, j, 0] = voxels[i, j, 0] + 1 # number of point
        voxels[i, j, 1] = (voxels[i, j, 1] * (voxels[i, j, 0] - 1) + radius ) / voxels[i, j, 0]  # average radius
        voxels[i, j, 2] = (voxels[i, j, 2] * (voxels[i, j, 0] - 1) + reflectance ) / voxels[i, j, 0] # average reflectance

        lats.append(latitude)

    bound = (min(lats),max(lats))
    
    return voxels,bound
    

def anti_globe_voxelization(globe:np.array, param:Param)->np.array:
    '''
    TODO: Never Debug
    input: globe latitude_n * longitude_n * 3
        per pixel : [number of points, average radius of points, average reflected reflectance of points]

    output: 
            points cloud n*[x,y,z,i]

    latitude : Angle between Point with Z-Positive [-PI/2, PI/2]
    longitude : Angle on XY-Plane (-PI, PI]

    '''
    pc = []

    min_bound = np.array([param.latitude_bound[0], param.longitude_bound[0]])
    max_bound = np.array([param.latitude_bound[1], param.longitude_bound[1]])
    resolution = (max_bound - min_bound)/np.array([param.latitude_n, param.longitude_n], dtype=float)

    for i,j in itertools.product(range(param.latitude_n),range(param.longitude_n)):
        latitude = param.min_bound[0] + i * param.resolution[0]
        longitude = param.min_bound[1] + j * param.resolution[1]
        radius = globe[i,j,1]
        reflectance = globe[i,j,2]
        point = np.array([latitude,longitude,radius,reflectance], dtype=float)
        pc.append(point)

    pc = np.stack(pc, axis=0)
    return pc

def merge(weather_entities, origin, param):
    '''
    TODO: Never Debug
    input: globe latitude_n * longitude_n * 3
        per pixel : [number of points, average radius of points, average reflected reflectance of points]
           origin n*[latitude, longitude, radius, reflectance]

    output: 
            points cloud n*[latitude, longitude, radius, reflectance]

    latitude : Angle between Point with Z-Positive [-PI/2, PI/2]
    longitude : Angle on XY-Plane (-PI, PI]
    '''
    batch_size, number_of_point, _ = origin.size()
    # batch_size, latitude_n,longitude_n, _ = weather_entities.size()
    origin = origin.detach().numpy()
    final = []


    for i in range(number_of_point):
        point = origin[:, i, :]
        latitude = point[:, 0]
        longitude = point[:, 1]

        i = ((latitude-param.min_bound[0]) / param.resolution[0]).astype(int)
        j = ((latitude-param.min_bound[1]) / param.resolution[1]).astype(int)

        i = np.clip(i, a_min=None, a_max=param.latitude_n-1)
        j = np.clip(j, a_min=None, a_max=param.longitude_n-1)

        latitude_entity = param.min_bound[0] + (i+np.random.uniform(-0.5, 0.5)) * param.resolution[0]
        longitude_entity = param.min_bound[1] + (j+np.random.uniform(-0.5, 0.5)) * param.resolution[1]

        loc = np.vstack((latitude, longitude)).T
        loc_entity = np.vstack((latitude_entity, longitude_entity)).T

        x = loc-loc_entity
        x = x**2
        x = np.sum(x, axis=1)
        x = np.sqrt(x)

        x = torch.from_numpy(x) # tensor
        size_entity = weather_entities[:,i,j,3] # tensor
        x = x/size_entity
        probs = torch.distributions.Normal(0, 1).cdf(x)
        random_numbers = torch.rand_like(probs)
        binary_tensor = (random_numbers <= probs).float()
        point = binary_tensor * torch.from_numpy(point)

        final.append(point)

        entity = np.column_stack(latitude_entity, longitude_entity, weather_entities[:,i,j,2], weather_entities[:,i,j,3])
        entity = (1-binary_tensor) * entity
        final.append(point)


        # radius = weather_entities[i,j,1]
        # reflectance = weather_entities[i,j,2]

    pass



def read_pcd(file, dataset = 'SeeingThroughFog' ): # TODO：to STF
    # pcd --> numpy.ndarrray
    # [...,...,[x,y,z,i,rgb],...,...] dtype = float
    pointcloud = []

    if (os.path.isfile(file)):
        if dataset == 'SeeingThroughFog':
            pointcloud = np.fromfile(file, dtype=np.float32)
            try:
                pointcloud = pointcloud.reshape((-1, 5))
            except Exception:
                pointcloud = pointcloud.reshape((-1, 4))
            pointcloud = pointcloud[:,0:4]
            # print(pointcloud)
        elif dataset == 'ADUULM':
            line_number = 0
            pointcloud = []
            # open file
            image_file = open(file,"r")
            for line in image_file:
                line_number = line_number + 1
                if line_number > 11: # Point: x,y,z,rgb=0,i
                    try:
                        buff = line.split()

                        x = float(buff[0])
                        y = float(buff[1])
                        z = float(buff[2])

                        i = float(buff[4])
                        rgb = float(buff[5])
                        
                        point = [x,y,z] 
                        pointcloud.append(point)
                    except IndexError:
                        print('IndexError')
                        print(file)
    else:
        print_warning('Not Found '+str(file))

    pointcloud = np.array(pointcloud)


    return pointcloud # as ndarray n*5 [x,y,z,i,rgb]


if __name__ == '__main__':
    LIDAR_NAME = 'lidar_vlp32_strongest' # lidar_vlp32_strongest, lidar_hdl64_strongest(10.-30)
    STF_PATH = Path('I:\Datasets\DENSE\SeeingThroughFog')
    PATH_TO_PARAM = Path('./utils/lidar_param.json')
    PATH_TO_LIDAR = STF_PATH.joinpath(LIDAR_NAME)

    WEIGHT,HEIGHT = 512, 256

    Path('./temp/').mkdir(exist_ok = True)
    videowriter = cv2.VideoWriter('./temp/globe_'+LIDAR_NAME+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps = 10, frameSize=(WEIGHT, HEIGHT))

    if STF_PATH.is_dir():
        files = list(PATH_TO_LIDAR.glob('*.bin'))
        files.sort(key=str)
    else:
        print_warning('Not Found '+str(STF_PATH))
        sys.exit()

    
    param = Param(LIDAR_NAME, PATH_TO_PARAM)

    print('lidar name:', LIDAR_NAME)
    bar = enumerate(files)
    bar = tqdm(bar, desc="Processing", total=len(files))

    max_bound = []
    min_bound = []

    try:
        for _, file in bar:
            bar.desc = str(file.stem)
            pc = read_pcd(file)
            # print(pc.shape)
            frame, bound = globe_voxelization(pc, param=param)
            frame = ndarray2img(frame)
            cv2.putText(img=frame, text=file.stem, org=(0, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 255, 255), thickness=2)
            videowriter.write(frame)
            # cv2.imshow('voxels', frame)
            # cv2.waitKey(0)
            max_bound.append(bound[1])
            min_bound.append(bound[0])


    except KeyboardInterrupt:
        pass

    videowriter.release()

    print(np.mean(max_bound)/PI*180)
    print(np.mean(min_bound)/PI*180)

    print(np.std(max_bound)/PI*180)
    print(np.std(min_bound)/PI*180)

    

    






    



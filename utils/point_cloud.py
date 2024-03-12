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

def lin2ham(pc:np.array)->np.array:
    '''
    Linear: [x,y,z,i]*n
    Height-Angle Matrix: 
    [
    [(i,r)]*how many of Heights }
    []                          } how many of Angles
    []                          }
    ...
    ]
    '''
    # [x,y,z,i] ==> [h,longitude,d,i]
    # pc[:,0:3] = Cart2Cylin(x=pc[:,0],y=pc[:,1],z=pc[:,2]) 
    pc[:,0:3] = Cart2Cylin(x=pc[:,0],y=pc[:,1],z=pc[:,2]) 


    number_of_not_zero = 0
    points_dict = {}
    # heights = []
    # angles = []

    for p in pc:
        h = p[0]
        t = p[1]
        d = p[2]
        i = p[3]

        h = np.round(h, 3)
        t = np.round(t, 2)
        if (h,t) not in points_dict.keys():
            points_dict[(h,t)] = [d,i]
            # heights.append(h)
            # angles.append(t)
        else:
            d_ = points_dict[(h,t)][0]
            if d <= d_ :
                points_dict[(h,t)] = [d,i]
                # heights.append(h)
                # angles.append(t)
            else:
                pass


    heights = np.sort(np.unique([key[0] for key in points_dict.keys()]))
    angles = np.sort(np.unique([key[1] for key in points_dict.keys()]))

    ham = np.zeros((len(heights), len(angles), 2))


    for i,j in itertools.product(range(len(heights)), range(len(angles))):
        try:
            ham[i,j] = points_dict[(heights[i],angles[j])]
            number_of_not_zero+=1
        except KeyError:
            pass

    # print('n=',num)
    return ham, heights, angles, number_of_not_zero


def globe_voxelization(pc:np.array, latitude_n:int=256, longitude_n:int=512, latitude_bound=[-30/180*PI,10/108*PI], longitude_bound=[-PI,PI]) ->np.array:
    '''
    input: points cloud n*[x,y,z,i]

    output: latitude_n * longitude_n * 3
        per pixel : [number of points, average radius of points, average reflected reflectance of points]

    latitude : Angle between Point with Z-Positive [-PI/2, PI/2]
    longitude : Angle on XY-Plane (-PI, PI]

    '''
    pc[:,0:3] = Cart2Spher(x=pc[:,0],y=pc[:,1],z=pc[:,2])
    # angles = pc[:, 0:2]

    min_bound = np.array([latitude_bound[0], longitude_bound[0]])
    max_bound = np.array([latitude_bound[1], longitude_bound[1]])

    voxels = np.zeros((latitude_n, longitude_n, 3), dtype=float)

    resolution = (max_bound - min_bound)/np.array([latitude_n, longitude_n], dtype=float)

    for p in pc:
        latitude = p[0]
        longitude = p[1]
        radius = p[2]
        reflectance = p[3]
        # print(p)
        i = int((latitude-min_bound[0]) / resolution[0])
        j = int((longitude-min_bound[1]) / resolution[1])

        i = min(latitude_n-1, i)
        j = min(longitude_n-1, j)

        voxels[i, j, 0] = voxels[i, j, 0] + 1 # number of point
        voxels[i, j, 1] = (voxels[i, j, 1] * (voxels[i, j, 0] - 1) + radius ) / voxels[i, j, 0]  # average radius
        voxels[i, j, 2] = (voxels[i, j, 2] * (voxels[i, j, 0] - 1) + reflectance ) / voxels[i, j, 0] # average reflectance
    
    return voxels
    

def anti_globe_voxelization(globe:np.array, latitude_n:int=256, longitude_n:int=512, latitude_bound=[-30/180*PI,10/108*PI], longitude_bound=[-PI,PI])->np.array:
    '''
    TODO: Never Debug
    input: globe latitude_n * longitude_n * 3
        per pixel : [number of points, average radius of points, average reflected reflectance of points]

    output: 
            points cloud n*[x,y,z,i]

    latitude : Angle between Point with Z-Positive [-PI/2, PI/2]
    longitude : Angle on XY-Plane (-PI, PI]

    '''
    latitude_n, longitude_n, _= globe.shape
    pc = []

    min_bound = np.array([latitude_bound[0], longitude_bound[0]])
    max_bound = np.array([latitude_bound[1], longitude_bound[1]])
    resolution = (max_bound - min_bound)/np.array([latitude_n, longitude_n], dtype=float)

    for i,j in itertools.product(range(latitude_n),range(longitude_n)):
        latitude = min_bound[0] + i * resolution[0]
        longitude = min_bound[1] + j * resolution[1]
        radius = globe[i,j,1]
        reflectance = globe[i,j,2]
        point = np.array([latitude,longitude,radius,reflectance], dtype=float)
        pc.append(point)

    pc = np.stack(pc, axis=0)
    return pc



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
    PATH_TO_LIDAR = Path('I:\Datasets\DENSE\SeeingThroughFog\lidar_hdl64_strongest')

    STF_PATH = Path('I:\Datasets\DENSE\SeeingThroughFog')

    WEIGHT,HEIGHT = 1024, 512

    Path('./temp/').mkdir(exist_ok = True)
    videowriter = cv2.VideoWriter('./temp/point_cloud_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps = 10, frameSize=(WEIGHT, HEIGHT))

    if STF_PATH.is_dir():
        files = list(STF_PATH.joinpath('lidar_hdl64_strongest').glob('*.bin'))
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
            # print(pc.shape)
            frame = globe_voxelization(pc, HEIGHT, WEIGHT)
            frame = ndarray2img(frame)
            cv2.putText(img=frame, text=file.stem, org=(0, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2)
            videowriter.write(frame)
            # cv2.imshow('voxels', frame)
            # cv2.waitKey(0)

    except KeyboardInterrupt:
        pass

    videowriter.release()
    

    






    



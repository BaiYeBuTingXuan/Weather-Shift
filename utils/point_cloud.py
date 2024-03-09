import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
import numpy as np

from pathlib import Path

from utils import print_warning,ndarray2img
from utils.math import Cart2Cylin,approx_equal,Cart2Spher

import cv2

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
    # [x,y,z,i] ==> [h,theta,d,i]
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


def directional_voxelization(pc:np.array, phi_n:int=256, theta_n:int=512, is_debug:bool=False) ->np.array: #  TODO: definite the bound of phi
    '''
    input: n*[x,y,z,i]

    output: phi_n * theta_n * 3
        per pixel : [number of points, average distance of points, average reflected intensity of points]

    Phi : Angle between Point with Z-Positive [0, PI]
    theta : Angle on XY-Plane (-PI, PI]

    '''
    pc[:,0:3] = Cart2Spher(x=pc[:,0],y=pc[:,1],z=pc[:,2])
    angles = pc[:, 0:2]

    min_bound = np.min(angles, axis=0)
    max_bound = np.max(angles, axis=0)

    voxels = np.zeros((phi_n, theta_n, 3), dtype=float)

    resolution = (max_bound - min_bound)/np.array([phi_n-1, theta_n-1], dtype=float)

    # print(min_bound)
    # print(max_bound)
    # print(resolution)

    for p in pc:
        phi = p[0]
        theta = p[1]
        distance = p[2]
        intensity = p[3]
        # print(p)
        i = int((phi-min_bound[0]) / resolution[0])
        j = int((theta-min_bound[1]) / resolution[1])
        
        voxels[i, j, 0] = voxels[i, j, 0] + 1 # number of point
        voxels[i, j, 1] = (voxels[i, j, 1] * (voxels[i, j, 0] - 1) + distance ) / voxels[i, j, 0]  # average distance
        voxels[i, j, 2] = (voxels[i, j, 2] * (voxels[i, j, 0] - 1) + intensity ) / voxels[i, j, 0] # average intensity
    
    if is_debug:
        return {'voxels': voxels, 'min_bound': min_bound, 'max_bound': max_bound, 'resolution': resolution}
    else:
        return voxels


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
    path = Path('I:\Datasets\DENSE\SeeingThroughFog\lidar_hdl64_strongest')/'2018-02-03_20-48-35_00400.bin'
    pc = read_pcd(path)
    print(pc.shape)
    total = pc.shape[0]
    v = directional_voxelization(pc, 256, 512)
    print(v.shape)
    np.min(v,axis=2)

    v = ndarray2img(v)
    # v = ((v - np.min(v, axis=(0, 1))) / np.max(v, axis=(0, 1))) * 255
    # v = v.astype(np.uint8)

    # 显示图片
    cv2.imshow('voxel', v)

    # 等待按键事件，关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()





    



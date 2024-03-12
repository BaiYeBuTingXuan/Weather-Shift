import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
import numpy as np
from utils import print_warning

PI = 3.14159265359

def angle_normal(angle):
    while angle >= np.pi:
        angle -= 2*np.pi
    while angle <= -np.pi:
        angle += 2*np.pi
    return angle


def spin(xy, deg, deg_OR_rad):
    x,y = xy
    if deg_OR_rad == 'deg':
        rad = deg/180*np.pi
    elif deg_OR_rad == 'rad':
        pass
    else:
        print_warning('Please input \'deg\' or \'rad\'')
        return 0

    # rotation equation:
    # [x] = [cost -sint][x']
    # [y] = [sint  cost][y']
    
    x_ = x*np.cos(rad)-y*np.sin(rad)
    y_ = x*np.sin(rad)+y*np.cos(rad)
    # res = ()
    return [x_,y_]


def sign(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    else:
        return 0
    

def Cart2Cylin(x,y,z): # Cartesian(x,y,z) to Cylinder(d,theta,h)
    radius = np.sqrt(np.square(x)+ np.square(y))  
    longitude = np.arctan2(x,y)
    height = z

    return np.stack([height,longitude,radius], axis=0).transpose()


def Cart2Spher(x, y, z):
    '''
    radius is the radial distance from the origin,
    longitude θ is the polar angle (angle in the xy-plane),
    latitude φ is the azimuthal angle (angle from the positive z-axis).
    '''
    radius = np.sqrt(np.square(x)+ np.square(y)) 
    latitude = np.pi/2 - np.arccos(z / radius)
    longitude = np.arctan2(y, x)

    return np.stack([latitude,longitude,radius], axis=0).transpose()


def approx_equal(a, b, epsilon=5e-3):
    return abs(a - b) <= epsilon


def deg2rad(deg:float):
    rad = deg/180*PI
    return rad


def rad2deg(rad:float):
    deg = rad/PI*180
    return deg
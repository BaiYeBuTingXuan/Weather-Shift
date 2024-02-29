import imp


import numpy as np

def angle_normal(angle):
    while angle >= np.pi:
        angle -= 2*np.pi
    while angle <= -np.pi:
        angle += 2*np.pi
    return angle

def spin(xy,deg):
    x,y = xy
    rad = deg/180*np.pi

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
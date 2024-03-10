import numpy as np

PI = 3.14159265359

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
    
def Cart2Cylin(x,y,z): # Cartesian(x,y,z) to Cylinder(d,theta,h)
    distance = np.sqrt(np.square(x)+ np.square(y))  
    theta = np.arctan2(x,y)
    height = z

    return np.stack([height,theta,distance], axis=0).transpose()



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
    """
    判断两个浮点数近似相等的函数

    Parameters:
    a (float): 第一个浮点数
    b (float): 第二个浮点数
    epsilon (float, optional): 允许的误差范围，默认为1e-9

    Returns:
    bool: 如果两个浮点数在给定的误差范围内近似相等，则返回True；否则返回False
    """
    return abs(a - b) <= epsilon




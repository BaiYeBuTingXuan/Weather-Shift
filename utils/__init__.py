import numpy as np
import argparse
import os
import sys  
sys.path.append('/home/wanghejun/Desktop/wanghejun/WeatherShift/Weather-Shift')
import glob
import time

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

def deepcopy_generator(original_generator):
    for item in original_generator:
        yield item


def print_warning(message):
    print('\033[93m [WARNING] ' + message + '\033[0m')


def ndarray2img(array):
    assert len(array.shape) == 3 # Only for 3-Dimension Tensor
    assert array.shape[2] == 3 or array.shape[2] == 1 # Only for RGB or Gray img that 3rd dimension == 3 or 1

    array = ((array - np.min(array, axis=(0, 1))) / np.max(array, axis=(0, 1))) * 255
    img = array.astype(np.uint8)

    return img


def write_params(log_path, parser, description=None):
    opt = parser.parse_args()
    options = parser._optionals._actions
    with open(log_path+'params.md', 'w+') as file:
        file.write('# Params\n')
        file.write('********************************\n')
        file.write('Time: '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n')
        if description is not None:
            file.write('**Description**: '+description+'\n')
        
        file.write('| Param | Value | Description |\n')
        file.write('| ----- | ----- | ----------- |\n')
        for i in range(len(parser._optionals._actions)):
            option = options[i]
            if option.dest != 'help':
                file.write('|**'+ option.dest+'**|'+str(opt.__dict__[option.dest])+'|'+option.help+'|\n')
        file.write('********************************\n\n')
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
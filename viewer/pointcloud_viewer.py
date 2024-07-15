import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
import numpy as np

from pathlib import Path

from utils import print_warning,ndarray2img
from utils.math_ import Cart2Cylin,approx_equal,Cart2Spher,Spher2Cart,PI

import cv2
from tqdm import tqdm
import itertools
import torch
import json

import copy
import gzip
import socket
import pandas
import logging
import argparse

import numpy as np
import pickle as pkl
import matplotlib as mpl
import matplotlib.cm as cm
import multiprocessing as mp
import pyqtgraph.opengl as gl

from glob import glob
from typing import List
from pathlib import Path
from plyfile import PlyData

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from pyqtgraph.Qt import QtGui



from utils.point_cloud import read_pcd


parser = argparse.ArgumentParser()
parser.add_argument('-b', '--base', type=str, help='path to where the file are',
                    default='/home/wanghejun/Desktop/wanghejun/WeatherShift/main')
parser.add_argument('-e', '--experiments', type=str, help='path to where you store your OpenPCDet experiments',
                    default=str(Path.home() / 'repositories/PCDet/output'))
args = parser.parse_args()

COLORS = {
    'origin'      : [255,   0,   0, 255], # origin in RED
    'traditional' : [  0, 255,   0, 255], # traditional in green
    'ours'        : [255, 255,   0, 255], # ours in BLUE

}

BASE_ROOT = Path(args.base)
DATASETS_ROOT = BASE_ROOT.joinpath('data')
EXPERIMENTS_ROOT = Path(args.experiments)

DENSE =  DATASETS_ROOT / 'Dense/SeeingThroughFog'

ORIGIN_CLOUD = DENSE / 'cloud/lidar_hdl64_strongest'
TRADITIONAL = DENSE / 'traditional/lidar_hdl64_strongest'
OURS = DENSE / 'dense_foggy_day/lidar_hdl64_strongest'
POINT_CLOUD_PATHs = {
    'origin'      : ORIGIN_CLOUD,
    'traditional' : TRADITIONAL,
    'ours'        : OURS       ,
}

PATH_TO_LIST = BASE_ROOT.joinpath('splits/test/clear_day.txt')

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

FILES_LIST = []
with open(PATH_TO_LIST,'r') as file:
    FILES_LIST = [l.strip().replace(',','_')+'.bin' for l in file.readlines ()]
    FILES_LIST.sort()
MAX_INDEX = len(FILES_LIST)-1

if socket.gethostname() == 'beast':
    DENSE = Path.home() / 'datasets_local' / 'DENSE/SeeingThroughFog/lidar_hdl64_strongest'

MIN_DISTANCE = 1.75
POINT_SIZE = 3


class MyWindow(QMainWindow):

    def __init__(self) -> None:

        super(MyWindow, self).__init__()

        self.index = -1
        self.current_pc = None

        self.pc_dict = {
                'origin'      : None,
                'traditional' : None,
                'ours'        : None,
        }

        self.load_origin_flag = False
        self.load_traditional_flag = False
        self.load_ours_flag = False

        self._set_monitor()
        self._set_Widget()

    def _set_monitor(self):
        hostname = socket.gethostname()
        if hostname == 'beast':
            self.monitor = QDesktopWidget().screenGeometry(1)
            self.monitor.setHeight(int(0.45 * self.monitor.height()))
        elif hostname == 'hox':
            self.monitor = QDesktopWidget().screenGeometry(2)
            self.monitor.setHeight(int(0.45 * self.monitor.height()))
        else:
            self.monitor = QDesktopWidget().screenGeometry(0)
            self.monitor.setHeight(self.monitor.height())
        self.setGeometry(self.monitor)

    def _set_Widget(self):
        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)

        self.layout = QGridLayout()
        self.centerWidget.setLayout(self.layout)

        self.grid_dimensions = 20
        self.viewer = gl.GLViewWidget()
        self.viewer.setWindowTitle('drag & drop point cloud viewer')
        self.viewer.setCameraPosition(distance=2 * self.grid_dimensions)
        self.layout.addWidget(self.viewer, 0, 0, 1, 6)

        self.grid = gl.GLGridItem()
        self.grid.setSize(self.grid_dimensions, self.grid_dimensions)
        self.grid.setSpacing(1, 1)
        self.grid.translate(0, 0, -2)
        self.viewer.addItem(self.grid)

        ###############
        # Bottom:reset
        ###############
        self.reset_btn = QPushButton("reset")
        self.reset_btn.clicked.connect(self.reset)
        self.layout.addWidget(self.reset_btn, 1, 5)
        ###############
        # Bottom:Origin
        ###############
        self.load_origin_btn = QPushButton("Origin")
        self.load_origin_btn.clicked.connect(self.load_origin)
        self.layout.addWidget(self.load_origin_btn, 1, 4)
        ###############
        # Bottom:Traditionl Simulate
        ###############
        self.load_traditional_btn = QPushButton("Traditionl Simulate")
        self.load_traditional_btn.clicked.connect(self.load_traditional)
        self.layout.addWidget(self.load_traditional_btn, 2, 4)
        ###############
        # Bottom:Our Simulate
        ###############
        self.load_our_btn = QPushButton("Our Simulate")
        self.load_our_btn.clicked.connect(self.load_ours)
        self.layout.addWidget(self.load_our_btn, 3, 4)

        self.select_btn = QPushButton("select an item of pointcloud")
        self.select_btn.clicked.connect(self.select_an_item)
        self.layout.addWidget(self.select_btn, 1, 1, 1, 2)

        self.prev_btn = QPushButton("<-")
        self.next_btn = QPushButton("->")

        self.prev_btn.clicked.connect(self.decrement_index)
        self.next_btn.clicked.connect(self.increment_index)

        self.layout.addWidget(self.prev_btn, 1, 0)
        self.layout.addWidget(self.next_btn, 1, 3)

    def show_pointcloud(self) -> None:
        self.clear_monitor()
        for key in self.pc_dict.keys():
            if self.pc_dict[key] is None:
                continue
            else:
                pc = self.pc_dict[key]
                colors = np.array([COLORS[key]]*pc.shape[0])# TODO

                mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=POINT_SIZE, color=colors)
                print('here')
                self.viewer.addItem(mesh)

    def set_enable(self, boolean:bool=False):
        self.reset_btn.setEnabled(boolean)
        self.next_btn.setEnabled(boolean)
        self.prev_btn.setEnabled(boolean)
    
    def clear_monitor(self):
        self.viewer.items = []
        self.viewer.addItem(self.grid)

    def reset(self):
        pass
    
    def load_origin(self):
        if self.load_origin_flag:
            self.pc_dict['origin'] = None
        else:
            if self.index == -1:
                pass
            else:
                self.load_pointcloud(key='origin')
                self.show_pointcloud()
        self.load_origin_flag = ~self.load_origin_flag

    def load_traditional(self):
        if self.load_traditional_flag:
            self.pc_dict['traditional'] = None
        else:
            if self.index == -1:
                pass
            else:
                self.load_pointcloud(key='traditional')
                self.show_pointcloud()
        self.load_traditional_flag = ~self.load_traditional_flag

    def load_ours(self):
        if self.load_ours_flag:
            self.pc_dict['ours'] = None
        else:
            if self.index == -1:
                pass
            else:
                self.load_pointcloud(key='ours')
                self.show_pointcloud()
        self.load_ours_flag = ~self.load_ours_flag
    
    def load_pointcloud(self, key):
        filename = POINT_CLOUD_PATHs[key].joinpath(FILES_LIST[self.index])
        pc = read_pcd(filename)
        self.pc_dict[key] = pc
        
    def select_an_item(self):
        selected, _ = QInputDialog.getItem(self, "Select Item", "Select an item:", FILES_LIST, editable=False)
        self.index = FILES_LIST.index(selected)
        
    def create_file_list(self, filename: str = None) -> None:

        if len(FILES_LIST) > 0:

            if filename is None:
                filename = self.FILES_LIST[0]

            self.index = self.get_index(filename)
            self.show_pointcloud(self.file_list[self.index])

    def decrement_index(self) -> None:
        if self.index != -1:
            self.index -= 1
            self.index = max(min(self.index,MAX_INDEX),0)
            self.update_index()

    def increment_index(self) -> None:
        if self.index != -1:
            self.index += 1
            self.index = max(min(self.index,MAX_INDEX),0)
            self.update_index()

    def update_index(self):
        for key in self.pc_dict.keys():
            if self.pc_dict[key] == None:
                pass
            else:
                self.load_pointcloud(key=key)
        

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.debug(pandas.__version__)

    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()
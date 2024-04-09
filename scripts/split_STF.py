import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--test_split', type=float, default=0.1, help='rate of test file')
parser.add_argument('--valid_split', type=float, default=0.1, help='rate of validation file')
parser.add_argument('--random_seed', type=int, default=6, help='random seed')
arg = parser.parse_args()

CURRENT_PATH = Path('.')
SAVE_PATH = CURRENT_PATH.joinpath('data/Dense/SeeingThroughFog/splits')
STF_LIST_PATH = Path('/home/wanghejun/Desktop/wanghejun/WeatherShift/main/data/Dense/SeeingThroughFog/splits/origin')
RNG = np.random.default_rng(seed=arg.random_seed)

assert arg.test_split >= 0 and arg.test_split <= 1, "test_split in [0,1]"

if __name__ == '__main__':
    SAVE_PATH.mkdir(exist_ok=True)
    SAVE_PATH.joinpath('test').mkdir(exist_ok=True)
    SAVE_PATH.joinpath('valid').mkdir(exist_ok=True)
    SAVE_PATH.joinpath('train').mkdir(exist_ok=True)


    files = list(STF_LIST_PATH.glob('*.txt'))
    bar = enumerate(files)
    bar = tqdm(bar, total=len(files))
    bar.desc = 'Processing'
    
    for _, file in bar:
        bar.desc = file.stem


        with open(file=file,mode="r") as f:
            lines = f.readlines()
            
        test_list = []
        test_len = int(len(lines)*arg.test_split)
        for _ in range(test_len):
            i = RNG.integers(low=0, high=len(lines)-1, size=1)[0]
            test_list.append(lines[i])
            lines.pop(i)

        valid_list = []
        valid_len = int(len(lines)*arg.valid_split)
        for _ in range(test_len):
            i = RNG.integers(low=0, high=len(lines)-1, size=1)[0]
            valid_list.append(lines[i])
            lines.pop(i)

        train_list = lines

        test_list.sort()
        valid_list.sort()
        train_list.sort()


        test_split = file.stem+'.txt'
        valid_split = file.stem+'.txt'
        train_split = file.stem+'.txt'


        with open(file=SAVE_PATH.joinpath('test', test_split), mode="w") as f:
            f.writelines(test_list)

        with open(file=SAVE_PATH.joinpath('valid', valid_split), mode="w") as f:
            f.writelines(valid_list)

        with open(file=SAVE_PATH.joinpath('train', train_split), mode="w") as f:
            f.writelines(train_list)
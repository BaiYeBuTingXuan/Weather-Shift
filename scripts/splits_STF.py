import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from utils import deepcopy_generator


parser = argparse.ArgumentParser()
parser.add_argument('--test_split', type=float, default=0.1, help='rate of test file')
parser.add_argument('--random_seed', type=int, default=6, help='random seed')
arg = parser.parse_args()

CURRENT_PATH = Path('.')
STF_PATH = Path('E:\WeatherSimulation\SeeingThroughFog')
RNG = np.random.default_rng(seed=arg.random_seed)

assert arg.test_split >= 0 and arg.test_split <= 1, "test_split in [0,1]"

if __name__ == '__main__':
    CURRENT_PATH.joinpath('splits').mkdir(exist_ok=True)

    files = list(STF_PATH.joinpath('splits').glob('*.txt'))
    bar = enumerate(files)
    bar = tqdm(bar, total=len(files))
    
    for _, file in bar:
        with open(file=file,mode="r") as f:
            lines = f.readlines()

        test_list = []
        test_len = int(len(lines)*arg.test_split)
        for _ in range(test_len):
            i = RNG.integers(low=0, high=len(lines)-1, size=1)[0]
            test_list.append(lines[i])
            lines.pop(i)

        train_list = lines

        test_split = file.stem+'_test'+'.txt'
        train_split = file.stem+'_train'+'.txt'

        with open(file=CURRENT_PATH.joinpath('splits', test_split),mode="w") as f:
            f.writelines(test_list)

        with open(file=CURRENT_PATH.joinpath('splits', train_split),mode="w") as f:
            f.writelines(train_list)
        

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

from pathlib import Path
import random
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from utils import mkdir, print_warning, write_params
from train.dataloader import SeeingThroughFogDataset, WEATHERS
from models import WeatherClassifier,weights_init
from models.unet import UNetGenerator


# seed = int(datetime.now().timestamp())
# random.seed(seed)
random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)


parser = argparse.ArgumentParser()
parser.add_argument('--train_name', type=str, default='0', help='list of train lidar')
parser.add_argument('--lidar', type=list, default=['lidar_hdl64_strongest'], help='list of train lidar')
parser.add_argument('--dataset_path', type=str, default="/home/wanghejun/Desktop/wanghejun/WeatherShift/main/data/Dense/SeeingThroughFog", help='path to the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--n_cpu', type=int, default=16, help='number of CPU threads to use during batches generating')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--train_time', type=int, default=1e3, help='total training time')
parser.add_argument('--checkpoints_interval', type=int, default=100, help='interval between model checkpoints')
parser.add_argument('--clip_value', type=float, default=1, help='Clip value for training to avoid gradient vanishing')
# parser.add_argument('--gamma', type=float, default=1, help='trade-off of L1 to the L2')
parser.add_argument('--target_weather', type=str, default='dense_fog_day', help='the weather that we need to simulate')
parser.add_argument('--path2clser', type=str, default="/home/wanghejun/Desktop/wanghejun/WeatherShift/main/data/Dense/SeeingThroughFog", help='path to the model of dicriminator')

opt = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:',device)

PATH_TO_GLOBE = Path(opt.dataset_path).joinpath('globe')
THIS_FILE_NAME = Path(__file__).stem

BASE_PATH = Path('/home/wanghejun/Desktop/wanghejun/WeatherShift/main')
RESULT_PATH = BASE_PATH.joinpath('result').joinpath(str(THIS_FILE_NAME)).joinpath(opt.train_name)
LOG_PATH = RESULT_PATH.joinpath('log')
SAVE_PATH = RESULT_PATH.joinpath('save')
SPLITS_PATH =BASE_PATH.joinpath('splits')

if not PATH_TO_GLOBE.is_dir():
    print_warning('Not Found '+str(PATH_TO_GLOBE))

LOG_PATH.mkdir(parents=True, exist_ok = True)
SAVE_PATH.mkdir(parents=True, exist_ok = True)

description = 'train '+THIS_FILE_NAME+' of '+opt.train_name
logger = SummaryWriter(log_dir=LOG_PATH)
write_params(str(LOG_PATH), parser, description)

""" generator model loading """
G = UNetGenerator().to(device)
G.apply(weights_init)

""" discriminator model loading """
D = WeatherClassifier().to(device)
PATH_TO_MODEL = Path(opt.path2clser)
if PATH_TO_MODEL.is_file():
    D.load_state_dict(torch.load(PATH_TO_MODEL))
else:
    D.apply(weights_init)
    print_warning('NOT FOUND model')
    print(opt.path2clser)

""" train dataset loading """
clear_day_train = DataLoader(SeeingThroughFogDataset(splits_path=str(SPLITS_PATH), dataset_path=PATH_TO_GLOBE, mode='train', weathers=['clear_day']), 
                                                batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

# dense_fog_train = DataLoader(SeeingThroughFogDataset(splits_path=str(SPLITS_PATH), dataset_path=PATH_TO_GLOBE, mode='train', weathers=['light_fog_day']), 
                                                # batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

""" validation dataset loading """
clear_day_valid = DataLoader(SeeingThroughFogDataset(splits_path=str(SPLITS_PATH), dataset_path=PATH_TO_GLOBE, mode='test', weathers=['clear_day']), 
                                                batch_size=1, shuffle=False, num_workers=1)

# dense_fog_valid = DataLoader(SeeingThroughFogDataset(splits_path=str(SPLITS_PATH), dataset_path=PATH_TO_GLOBE, mode='test', weathers=['light_fog_day']), 
                                                # batch_size=1, shuffle=False, num_workers=1)
# test_samples = iter(test_loader)

""" Loss and optimizer """
criterion = torch.nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(G.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)


""" label  """
label = WEATHERS.index(opt.target_weather)
label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=len(WEATHERS)+1).type(torch.float32).to(device)

def evaluate(total_step):
    G.eval()
    loss = []
    for i, batch in enumerate(clear_day_valid):
        batch['globe'] = batch['globe'].to(device)
        batch['globe'].requires_grad = True

        fake = G(batch['globe'])
        pred = D(fake)

        loss = criterion(pred, label)

        loss.append(criterion(pred, label).item())

    loss = np.array(loss)
    logger.add_scalar('test/loss',loss.mean(), total_step)
    G.train()



if __name__ == '__main__':
    total_step = 0

    for epoch in range(opt.epoch, opt.n_epochs):
        print(f"epoch: {epoch}")

        bar = enumerate(clear_day_train)
        length = len(clear_day_train)
        bar = tqdm(bar, total=length)
        for i, batch in bar:
            total_step += 1
            
            batch['globe'] = batch['globe'].to(device)
            # batch['weather'] = batch['weather'].to(device)
            batch['globe'].requires_grad = True

            fake = G(batch['globe'])
            pred = D(fake)

            loss = criterion(pred, label)

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_value_(G.parameters(), clip_value=opt.clip_value)

            optimizer.step()

            logger.add_scalar('train/loss',loss.item(), total_step)

            if total_step % opt.checkpoints_interval == 0:
                str(SAVE_PATH.joinpath('model_%d.pth' % total_step))
                torch.save(G.state_dict(), SAVE_PATH.joinpath('model_%d.pth' % total_step))
                evaluate(total_step)





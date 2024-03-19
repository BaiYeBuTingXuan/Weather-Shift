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
from train.dataloader import SeeingThroughFogDataset
from models import WeatherClassifier,weights_init

# seed = int(datetime.now().timestamp())
# random.seed(seed)
random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)


parser = argparse.ArgumentParser()
parser.add_argument('--train_name', type=str, default='1', help='list of train lidar')
parser.add_argument('--lidar', type=list, default=['lidar_hdl64_strongest'], help='list of train lidar')
parser.add_argument('--dataset_path', type=str, default="/home/wanghejun/Desktop/wanghejun/WeatherShift/main/data/Dense/SeeingThroughFog", help='path of the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--n_cpu', type=int, default=16, help='number of CPU threads to use during batches generating')
parser.add_argument('--lr', type=float, default=5e-6, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--train_time', type=int, default=1e3, help='total training time')
parser.add_argument('--checkpoints_interval', type=int, default=100, help='interval between model checkpoints')
parser.add_argument('--clip_value', type=float, default=1, help='Clip value for training to avoid gradient vanishing')
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

""" model loading """
model = WeatherClassifier().to(device)
model.apply(weights_init)

""" train dataset loading """
train_loader = DataLoader(SeeingThroughFogDataset(splits_path=str(SPLITS_PATH), dataset_path=PATH_TO_GLOBE, mode='train'), 
                                                batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

""" validation dataset loading """
test_loader_batch_size = 256
test_loader = DataLoader(SeeingThroughFogDataset(splits_path=str(SPLITS_PATH), dataset_path=PATH_TO_GLOBE, mode='valid'), 
                                                batch_size=test_loader_batch_size, shuffle=False, num_workers=1)
# test_samples = iter(test_loader)

""" Loss and optimizer """
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)


def evaluate(total_step):
    model.eval()
    loss = []
    
    correct_predictions = 0
    for i, batch in enumerate(test_loader):
        
        batch['globe'] = batch['globe'].to(device)
        batch['weather'] = batch['weather'].to(device)

        batch['globe'].requires_grad = False
        batch['weather'].requires_grad = False

        prediction = model(batch['globe'])

        pred = np.argmax(prediction.cpu().detach().numpy(), axis=1) 
        label = np.argmax(batch['weather'].cpu().detach().numpy(), axis=1) 
        correct_predictions += sum(pred == label)  


        loss.append(criterion(prediction, batch['weather']).item())

    loss = np.array(loss)
    logger.add_scalar('valid/loss',loss.mean(), total_step)

    accuracy = correct_predictions / ( len(test_loader)*test_loader_batch_size )
    # print(accuracy)
    logger.add_scalar('valid/accuracy', accuracy, total_step)
    model.train()

        

if __name__ == '__main__':
    total_step = 0

    for epoch in range(opt.epoch, opt.n_epochs):
        print(f"epoch: {epoch}")

        bar = enumerate(train_loader)
        length = len(train_loader)
        bar = tqdm(bar, total=length)

        for i, batch in bar:
            total_step += 1

            batch['globe'] = batch['globe'].to(device)
            batch['weather'] = batch['weather'].to(device)
            batch['globe'].requires_grad = True

            prediction = model(batch['globe'])

            loss = criterion(prediction, batch['weather'])

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=opt.clip_value)

            optimizer.step()

            logger.add_scalar('train/loss',loss.item(), total_step)

            if total_step % opt.checkpoints_interval == 0:
                str(SAVE_PATH.joinpath('model_%d.pth' % total_step))
                torch.save(model.state_dict(), SAVE_PATH.joinpath('model_%d.pth' % total_step))
                evaluate(total_step)





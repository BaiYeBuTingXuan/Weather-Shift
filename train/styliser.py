import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from train.dataloader import SeeingThroughFogDataset2, WEATHERS
from models import WeatherClassifier,weights_init
from models.unet import UNetGenerator_Normal as Generator


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
parser.add_argument('--lr', type=float, default=1e-9, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--train_time', type=int, default=1e3, help='total training time')
parser.add_argument('--checkpoints_interval', type=int, default=100, help='interval between model checkpoints')
parser.add_argument('--clip_value', type=float, default=0.1, help='Clip value for training to avoid gradient vanishing')
parser.add_argument('--gamma', type=float, default=1000, help='trade-off of total style loss to total content loss')
parser.add_argument('--iota', type=list, default=[1e-5,1,0.4], help='trade-off of items of style loss')
parser.add_argument('--target_weather', type=str, default='dense_fog_day', help='the weather that we need to simulate')
parser.add_argument('--path2clser', type=str, default="/home/wanghejun/Desktop/wanghejun/WeatherShift/main/models/trained/WeatherClassifier/C.pth", help='path to the model of dicriminator')

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
G = Generator(in_channels=5).to(device)
G.apply(weights_init)

""" discriminator model loading """
D = WeatherClassifier(in_channels=5).to(device)
PATH_TO_MODEL = Path(opt.path2clser)
if PATH_TO_MODEL.is_file():
    D.load_state_dict(torch.load(PATH_TO_MODEL))
else:
    D.apply(weights_init)
    print_warning('NOT FOUND model')
    print(opt.path2clser)

""" train dataset loading """
train_loader = DataLoader(SeeingThroughFogDataset2(splits_path=str(SPLITS_PATH), dataset_path=PATH_TO_GLOBE, mode='train'), 
                                                batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

""" validation dataset loading """
valid_batch_size=32
valid_loader = DataLoader(SeeingThroughFogDataset2(splits_path=str(SPLITS_PATH), dataset_path=PATH_TO_GLOBE, mode='valid'), 
                                                batch_size=valid_batch_size, shuffle=False, num_workers=1)
# test_samples = iter(test_loader)

""" Loss and optimizer """

# def criterion(pred_weathered, label_weathered):

CrossEntropy = torch.nn.CrossEntropyLoss().to(device)
L2 = torch.nn.L1Loss().to(device)


def loss_function(clear, weathered, synthesized):

    def gram(x):
        # b, c, h, w = x.size()
        # x = x.transpose(0,1,3,2)
        x = x-torch.mean(x, dim=(-2, -1), keepdim=True) # centerilize
        x_ = x.transpose(3,2) # centerilize
        return torch.matmul(x, x_)

    def style_loss(origin, synthesized):
        gram_origin = gram(origin)
        gram_synthesized = gram(synthesized)
        return L2(gram_origin, gram_synthesized)

    def content_loss(origin, weathered):
        return L2(origin,  weathered)


    x = torch.stack([clear, weathered, synthesized], dim=0)
    _, batch_size, color, height, width = x.size()
    x = x.reshape(-1, color, height ,width)
    D.eval()

    _, f = D(x)
    # f1, f2, f3, f4, f5 = f
    f_clear, f_weathered, f_synthesized = f[0][0:batch_size],f[0][batch_size:2*batch_size],f[0][2*batch_size:3*batch_size]
    content = content_loss(f_clear, f_synthesized)
    style_0 = style_loss(f_clear, f_weathered)

    f_clear, f_weathered, f_synthesized = f[1][0:batch_size],f[1][batch_size:2*batch_size],f[1][2*batch_size:3*batch_size]
    style_1 = style_loss(f_clear, f_weathered)

    f_clear, f_weathered, f_synthesized = f[2][0:batch_size],f[2][batch_size:2*batch_size],f[3][2*batch_size:3*batch_size]
    style_2 = style_loss(f_clear, f_weathered)

    # style = style_0+style_1+style_2 [0].unsqueeze(1)

    contents = [content]
    styles = [style_0, style_1, style_2]

    return contents,styles


optimizer = torch.optim.Adam(G.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)


""" label  """
# label_clear = WEATHERS.index('None')
# label_clear = torch.nn.functional.one_hot(torch.tensor(label_clear), num_classes=len(WEATHERS)).type(torch.float32).to(device)
# expanded_label = torch.stack([label_clear] * opt.batch_size, dim=0)

def evaluate(total_step):
    G.eval()
    loss_dict = {
        'total' : [],
        'content' : [],
        'style' : [],
        'style_0' : [],
        'style_1' : [],
        'style_2' : []
    }

    for _, batch in enumerate(valid_loader):
    
        clear = batch['source'].to(device)
        target = batch['target'].to(device)
        weather = batch['weather'].to(device)

        # print('here')

        clear.requires_grad = False
        target.requires_grad = False
        weather.requires_grad = False

        fake_weathered,_ = G(clear,weather)
        # pred_weathered, _ = D(fake_weathered)

        content_losses, sytle_losses = loss_function(clear, target, fake_weathered)
        content_0 = content_losses[0]
        style_0 = sytle_losses[0]
        style_1 = sytle_losses[1]
        style_2 = sytle_losses[2]

        content_loss = content_0
        sytle_loss = opt.iota[0]*style_0+opt.iota[1]*style_1+opt.iota[2]*style_2
        loss = content_loss + sytle_loss * opt.gamma

        loss_dict['total'].append(loss.item())
        loss_dict['content'].append(content_loss.item())
        loss_dict['style'].append(sytle_loss.item())
        loss_dict['style_0'].append(style_0.item())
        loss_dict['style_1'].append(style_1.item())
        loss_dict['style_2'].append(style_2.item())

    for key in loss_dict.keys():
        logger.add_scalar('valid/'+ key+ '_loss',np.array(loss_dict[key]).mean() , total_step)

    G.train()


if __name__ == '__main__':
    total_step = 0

    for epoch in range(opt.epoch, opt.n_epochs):
        print(f"epoch: {epoch}")

        bar = enumerate(train_loader)
        length = len(train_loader)
        bar = tqdm(bar, total=length)
        for i, batch in bar:
            total_step += 1
            
            clear = batch['source'].to(device)
            target = batch['target'].to(device)
            weather = batch['weather'].to(device)

            # print('here')
            clear.requires_grad = True
            target.requires_grad = False
            weather.requires_grad = True

            fake_weathered,_ = G(clear,weather)
            # pred_weathered, _ = D(fake_weathered)

            content_losses, sytle_losses = loss_function(clear, target, fake_weathered)
            content_0 = content_losses[0]
            style_0 = sytle_losses[0]
            style_1 = sytle_losses[1]
            style_2 = sytle_losses[2]

            content_loss = content_0
            sytle_loss = opt.iota[0]*style_0+opt.iota[1]*style_1+opt.iota[2]*style_2
            loss = content_loss + sytle_loss * opt.gamma

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_value_(G.parameters(), clip_value=opt.clip_value)

            optimizer.step()

            logger.add_scalar('train/total', loss.item(), total_step)
            logger.add_scalar('train/content', content_loss.item(), total_step)
            logger.add_scalar('train/style', sytle_loss.item(), total_step)
            logger.add_scalar('train/style_0', style_0.item(), total_step)
            logger.add_scalar('train/style_1', style_1.item(), total_step)
            logger.add_scalar('train/style_2', style_2.item(), total_step)


            if total_step % opt.checkpoints_interval == 0:
                str(SAVE_PATH.joinpath('model_%d.pth' % total_step))
                torch.save(G.state_dict(), SAVE_PATH.joinpath('model_%d.pth' % total_step))
                evaluate(total_step)





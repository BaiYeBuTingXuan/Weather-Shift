import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
from train.dataloader import SeeingThroughFogDataset, SeeingThroughFogDataset2, WEATHERS
from models import WeatherClassifier,WeatherClassifier2, weights_init
from models.unet import UNetGenerator_Normal,UNetGenerator_Normal2
from models.loss_func import InvExp_Loss


torch.autograd.set_detect_anomaly(True)
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
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--n_cpu', type=int, default=16, help='number of CPU threads to use during batches generating')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--train_time', type=int, default=1e3, help='total training time')
parser.add_argument('--checkpoints_interval', type=int, default=1000, help='interval between model checkpoints')
parser.add_argument('--clip_value', type=float, default=1, help='Clip value for training to avoid gradient vanishing')
parser.add_argument('--gamma1', type=float, default=1.717, help='trade-off of InvExp(Constant Loss) to the CEL(Weather Loss)')
parser.add_argument('--gamma2', type=float, default=4, help='trade-off of L1(Constant Loss) to the CEL(Weather Loss)')

parser.add_argument('--target_weather', type=str, default='dense_fog_day', help='the weather that we need to simulate')
parser.add_argument('--sigma', type=str, default=0.1, help='sigma of Bell loss')
# parser.add_argument('--lidar', type=list, default=['lidar_vlp32_strongest'], help='lidar to train(lidar_hdl64_strongest,lidar_vlp32_strongest)')


# parser.add_argument('--globe_type', type=str, default='globe_nor', help='globe or binormalizing globe')
parser.add_argument('--path2clser', type=str, default="/data2/wanghejun/WeatherShift/main/models/trained/WeatherClassifier/G_C.pth", help='path to the model of dicriminator')
# parser.add_argument('--path2clser', type=str, default="-", help='path to the model of dicriminator')

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

G_SAVE_PATH = SAVE_PATH.joinpath('G')
D_SAVE_PATH = SAVE_PATH.joinpath('D')

LOG_PATH.mkdir(parents=True, exist_ok = True)
G_SAVE_PATH.mkdir(parents=True, exist_ok = True)
D_SAVE_PATH.mkdir(parents=True, exist_ok = True)

description = 'train '+THIS_FILE_NAME+' of '+opt.train_name
logger = SummaryWriter(log_dir=LOG_PATH)
write_params(str(LOG_PATH), parser, description)

""" generator model loading """
G = UNetGenerator_Normal(in_channels=4).to(device)
# PATH_TO_MODEL = Path('/data2/wanghejun/WeatherShift/main/models/trained/Simulator/model_349000.pth')
PATH_TO_MODEL = Path('/data2/wanghejun/WeatherShift/main/models/trained/Simulator/model_200000.pth')
if PATH_TO_MODEL.is_file():
    G.load_state_dict(torch.load(PATH_TO_MODEL))
else:
    G.apply(weights_init)
    print_warning('NOT FOUND G model')

""" discriminator model loading """
D = WeatherClassifier2(in_channels=4).to(device)
# PATH_TO_MODEL = Path('/data2/wanghejun/WeatherShift/main/models/trained/WeatherClassifier/model_349000.pth')
PATH_TO_MODEL = Path('/data2/wanghejun/WeatherShift/main/models/trained/WeatherClassifier/model_200000.pth')
if PATH_TO_MODEL.is_file():
    D.load_state_dict(torch.load(PATH_TO_MODEL))
else:
    D.apply(weights_init)
    print_warning('NOT FOUND D model')

# x = torch.rand((32, 5, 128, 256), dtype=torch.float32).to(device)
# w = torch.rand([32, 10]).to(device)
# traced_model_G = torch.jit.trace(G, (x, w))
# # 将追踪后的模型添加到 TensorBoard 日志中
# logger.add_graph(traced_model_G, (x, w))

# traced_model_D = torch.jit.trace(D, x)
# # 将追踪后的模型添加到 TensorBoard 日志中
# logger.add_graph(traced_model_D, (x, w))

""" train dataset loading """
train_loader = DataLoader(SeeingThroughFogDataset2(splits_path=str(SPLITS_PATH), dataset_path=PATH_TO_GLOBE, mode='train', lidars=['lidar_hdl64_strongest']), 
                                                batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

""" validation dataset loading """
valid_batch_size=256
valid_loader = DataLoader(SeeingThroughFogDataset2(splits_path=str(SPLITS_PATH), dataset_path=PATH_TO_GLOBE, mode='valid', lidars=['lidar_hdl64_strongest']), 
                                                batch_size=valid_batch_size, shuffle=False, num_workers=1)


""" Loss and optimizer """

# def criterion(pred_weathered, label_weathered):

CrossEntropyLoss = torch.nn.CrossEntropyLoss().to(device)
InvExploss = InvExp_Loss(sigma=opt.sigma).to(device)
L1loss = torch.nn.MSELoss().to(device)

optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)



""" label  """
label_clear = WEATHERS.index('None')
label_clear = torch.nn.functional.one_hot(torch.tensor(label_clear), num_classes=len(WEATHERS)).type(torch.float32).to(device)
label_of_None = torch.stack([label_clear] * opt.batch_size, dim=0)

# print(label_of_None)

def evaluate(total_step):
    G.eval()
    D.eval()

    loss_dict = {
        'G_total' : [],
        'G_CrossEntropyLoss' : [],
        'G_InvExp' : [],
        'G_l1' : [],
        'D_total' : [],
        'D_fake' : [],
        'D_real' : []

    }
    # for i in range(100):
        # batch = valid_samples.next()
    for i, batch in enumerate(valid_loader):
        clear = batch['source'].to(device)
        weathered =batch['target'].to(device)
        label_weathered = batch['weather'].to(device)
        # print('here')
        # print(label_weathered[0])

        clear.requires_grad = False
        label_weathered.requires_grad = False

        fake_weathered,extra = G(clear,label_weathered)
        pred_fake, _ = D(fake_weathered)
        pred_real, _ = D(weathered)
        alpha = extra[0]
        radius = extra[1]
        cel = CrossEntropyLoss(pred_fake, label_weathered)
        ie =  InvExploss(radius)
        l1 = L1loss(alpha,torch.zeros_like(alpha))
        loss_G = cel + ie * opt.gamma1 + l1 * opt.gamma2

        label_of_None_for_test = torch.stack([label_clear] * pred_fake.size()[0], dim=0)


        fake = CrossEntropyLoss(pred_fake, label_of_None_for_test)
        real = CrossEntropyLoss(pred_real, label_weathered)
        loss_D = fake + real

        loss_dict['G_total'].append(loss_G.item())
        loss_dict['G_CrossEntropyLoss'].append(cel.item())
        loss_dict['G_InvExp'].append(ie.item())
        loss_dict['G_l1'].append(l1.item())

        loss_dict['D_total'].append(loss_D.item())
        loss_dict['D_fake'].append(fake.item())
        loss_dict['D_real'].append(real.item())


    for key in loss_dict.keys():
        value = np.array(loss_dict[key]).mean()
        logger.add_scalar('valid/'+key,value, total_step)
    
    
    alpha,radius,reflectance = extra
    mu = radius[:,0,:,:]
    var = radius[:,1,:,:]

    logger.add_histogram('valid/radius_mu', mu, total_step)
    logger.add_histogram('valid/radius_var', var, total_step)

    logger.add_histogram('valid/alpha', alpha, total_step)

    mu = reflectance[:,0,:,:]
    var = reflectance[:,1,:,:]
    logger.add_histogram('valid/reflectance_mu', mu, total_step)
    logger.add_histogram('valid/reflectance_var', var, total_step)

    G.train()
    D.train()

    

if __name__ == '__main__':
    total_step = -1

    for epoch in range(opt.epoch, opt.n_epochs):
        print(f"epoch: {epoch}")
        
        bar = enumerate(train_loader)
        length = len(train_loader)
        bar = tqdm(bar, total=length)
        for i, batch in bar:
            total_step += 1
            
            clear = batch['source'].to(device)
            weathered =batch['target'].to(device)
            label_weathered = batch['weather'].to(device)
            

            clear.requires_grad = True
            weathered.requires_grad = True
            label_weathered.requires_grad = True

            fake_weathered,extra = G(clear, label_weathered)
            pred_fake,_ = D(fake_weathered)

            alpha = extra[0]
            radius = extra[1]
            cel = CrossEntropyLoss(pred_fake, label_weathered)
            ie =  InvExploss(radius)
            l1 = L1loss(alpha,torch.zeros_like(alpha))
            loss_G = cel + ie * opt.gamma1 + l1 * opt.gamma2
            

            # loss = l1
            optimizer_G.zero_grad()

            loss_G.backward()

            torch.nn.utils.clip_grad_value_(G.parameters(), clip_value=opt.clip_value)

            optimizer_G.step()

            # print(ie.item())
            # print(cel.item())
            # print(opt.gamma1)
            # print(l1.item())
            # print(loss_G.item())
            # print('xx')

            logger.add_scalar('train/G/CrossEntropyLoss',cel.item(), total_step)
            logger.add_scalar('train/G/InvExpLoss',ie.item(), total_step)
            logger.add_scalar('train/G/l1',l1.item(), total_step)
            logger.add_scalar('train/G/loss',loss_G.item(), total_step)

            pred_fake,_ = D(fake_weathered.detach()) # OR  RuntimeError: Trying to backward through the graph a second time 
            pred_real,_ = D(weathered)

            fake = CrossEntropyLoss(pred_fake, label_of_None)
            real = CrossEntropyLoss(pred_real, label_weathered)
            loss_D = fake*(1/10) + real

            optimizer_D.zero_grad()

            loss_D.backward()

            torch.nn.utils.clip_grad_value_(D.parameters(), clip_value=opt.clip_value)

            optimizer_D.step()

            logger.add_scalar('train/D/fake_CrossEntropyLoss',fake.item(), total_step)
            logger.add_scalar('train/D/real_CrossEntropyLoss',real.item(), total_step)
            logger.add_scalar('train/D/total_CrossEntropyLoss',loss_D.item(), total_step)

            if total_step % opt.checkpoints_interval == 0:
                torch.save(G.state_dict(), G_SAVE_PATH.joinpath('model_%d.pth' % total_step))
                torch.save(D.state_dict(), D_SAVE_PATH.joinpath('model_%d.pth' % total_step))
                evaluate(total_step)





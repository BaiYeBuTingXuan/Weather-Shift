import os

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

from utils import write_params


random.seed(datetime.now())
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="img2path-01", help='name of the train')
parser.add_argument('--dataset_path', type=str, default="./dataset/", help='path of the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--n_cpu', type=int, default=16, help='number of CPU threads to use during batches generating')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--train_time', type=int, default=1e3, help='total training time')
parser.add_argument('--gamma', type=float, default=0.1, help='xy and vxy loss trade off')
parser.add_argument('--gamma2', type=float, default=0.001, help='xy and axy loss trade off')
parser.add_argument('--checkpoints_interval', type=int, default=2000, help='interval between model checkpoints')
parser.add_argument('--clip_value', type=float, default=1, help='Clip value for training to avoid gradient vanishing')
parser.add_argument('--adjust_dist', type=float, default=25., help='max distance')
parser.add_argument('--adjust_t', type=float, default=3., help='max time')
parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')

opt = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

description = 'train'
LOG_PATH = 'result/log/'+opt.dataset_name+'/'
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)

logger = SummaryWriter(log_dir=LOG_PATH)
write_params(LOG_PATH, parser, description)

""" model loading """
generator = None
discriminator = None

""" train dataset loading """
train_loader = DataLoader(ImgRouteDataset(data_index=[7], opt=opt), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

""" validation dataset loading """
test_loader = DataLoader(ImgRouteDataset(data_index=[1], opt=opt, evalmode=True), batch_size=1, shuffle=False, num_workers=1)
test_samples = iter(test_loader)  # 设置迭代器

""" Loss and optimizer """
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)


def evaluate(total_step):
    loss = []
    for i, batch in enumerate(test_loader):
        
        batch['point_cloud'] = batch['point_cloud'].to(device)

        batch['point_cloud'].requires_grad = False

        fake_pc = generator(batch['point_cloud'])

        category = discriminator(fake_pc)

        loss.append(criterion(category, batch['category']).item())

    logger.add_scalar('test/loss',loss.mean(), total_step)

        

if __name__ == '__main__':
    total_step = 0

    for epoch in range(opt.epoch, opt.n_epochs):
        print(f"epoch: {epoch}")

        bar = enumerate(train_loader)
        length = len(train_loader)
        bar = tqdm(bar, total=length)

        for i, batch in bar:
            total_step += 1

            batch['point_cloud'] = batch['point_cloud'].to(device)

            batch['point_cloud'].requires_grad = True

            fake_pc = generator(batch['point_cloud'])

            category = discriminator(fake_pc)

            loss = criterion(category, batch['category'])

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_value_(generator.parameters(), clip_value=opt.clip_value)

            optimizer.step()

            logger.add_scalar('train/loss',loss.item(), total_step)

            if total_step % opt.checkpoints_interval == 0:
                torch.save(generator.state_dict(), 'result/saved_models/%s/model_%d.pth' % (opt.dataset_name, total_step))
                evaluate(total_step)





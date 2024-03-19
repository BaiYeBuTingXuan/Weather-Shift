import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import argparse
import random



from train.dataloader import SeeingThroughFogDataset, WEATHERS
from models import WeatherClassifier,weights_init


# seed = int(datetime.now().timestamp())
# random.seed(seed)
random.seed(666)
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)

parser = argparse.ArgumentParser()
parser.add_argument('--lidar', type=list, default=['lidar_hdl64_strongest'], help='list of train lidar')
parser.add_argument('--dataset_path', type=str, default="/home/wanghejun/Desktop/wanghejun/WeatherShift/main/data/Dense/SeeingThroughFog", help='path to the dataset')
parser.add_argument('--save_epoch', type=int, default=220400, help='saved epoch of the model for test')
parser.add_argument('--model_path', type=str, default="/home/wanghejun/Desktop/wanghejun/WeatherShift/main/result/weather_classifier/0/save", help='path to the trained model')

opt = parser.parse_args()

PATH_TO_GLOBE = Path(opt.dataset_path).joinpath('globe')
PATH_TO_MODEL = Path(opt.model_path).joinpath('model_'+str(opt.save_epoch)+'.pth')
BASE_PATH = Path('/home/wanghejun/Desktop/wanghejun/WeatherShift/main')
SPLITS_PATH =BASE_PATH.joinpath('splits')

dataset = SeeingThroughFogDataset(lidars = opt.lidar, splits_path=str(SPLITS_PATH), dataset_path=PATH_TO_GLOBE, mode='test')
test_loader = DataLoader(dataset,batch_size=128, shuffle=False, num_workers=1)
total_samples = len(dataset)  
correct_predictions = 0 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:',device)

model = WeatherClassifier().to(device)
print('model:', PATH_TO_MODEL)
model.load_state_dict(torch.load(PATH_TO_MODEL))

bar = enumerate(test_loader)
length = len(test_loader)
bar = tqdm(bar, total=length)

labels = []
preds = []

for i, batch in bar:
    input = batch['globe'].to(device)
    label = batch['weather'].to(device)
    input.requires_grad = False


    outputs = model(input)  
    pred = np.argmax(outputs.cpu().detach().numpy(), axis=1) 
    label = np.argmax(label.cpu().detach().numpy(), axis=1) 
    # print(pred)
    # print(label)

    correct_predictions += sum(pred == label)  

    labels = labels + label.tolist()
    preds = preds + pred.tolist()

accuracy = correct_predictions / total_samples
print('Accuracy =', accuracy)
accuracy = np.round(accuracy, 3)

# print(labels)
# print(preds)
conf_matrix = confusion_matrix(preds, labels)

normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

print("Confusion Matrix:")
print(normalized_conf_matrix)

# 假设 conf_matrix 是混淆矩阵
# 你可以使用上文提到的 confusion_matrix 函数来获取
# 这里假设 conf_matrix 是一个 3x3 的混淆矩阵
# conf_matrix = np.array([[10, 2, 3],
                        # [1, 15, 2],
                        # [0, 2, 20]])

# 绘制热力图
plt.figure(figsize=(20, 20))
# sns.heatmap(normalized_conf_matrix, annot=True, cmap='Blues', fmt='g')
sns.heatmap(normalized_conf_matrix, annot=False, cmap='Reds', fmt='g')
#


# 设置标题和标签
plt.title('Confusion Matrix(Accuracy={accuracy})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.xticks(np.arange(len(WEATHERS)) + 0.5, WEATHERS)
plt.yticks(np.arange(len(WEATHERS)) + 0.5, WEATHERS)
plt.savefig('confusion_matrix_heatmap.png')

# 显示热力图
# plt.show()


import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils_pred import DataLoader
from model.Prediction import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob
from matplotlib import pyplot as plt
import time

import argparse


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='anomaly-score weight')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model')
parser.add_argument('--m_items_dir', type=str, help='directory of model')
parser.add_argument('--score_output', type=str, default=None, help='path used to save per-frame anomaly scores')
parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto', help='evaluation device')

args = parser.parse_args()

torch.manual_seed(2020)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpus)

torch.backends.cudnn.enabled = True
if args.device == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(args.device)
if device.type == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but is unavailable. Use --device cpu or install a CUDA PyTorch build.')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
print('Using device:', device)

test_folder = os.path.join(args.dataset_path, args.dataset_type, "testing", "frames")

test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')

if not args.model_dir or not args.m_items_dir:
    parser.error('--model_dir and --m_items_dir are required')
model = torch_load_compat(args.model_dir, map_location=device)
model.to(device)
m_items = torch_load_compat(args.m_items_dir, map_location=device)


labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')
if args.dataset_type == 'shanghaitech':
    labels = np.expand_dims(labels, 0)

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
for video in videos_list:
    video_name = os.path.basename(video)
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}
labels_by_video = {}

print('Evaluation of', args.dataset_type)

for video in sorted(videos_list):
    video_name = os.path.basename(video)
    video_labels = labels[0][4+label_length:videos[video_name]['length']+label_length]
    labels_list = np.append(labels_list, video_labels)
    labels_by_video[video_name] = np.asarray(video_labels, dtype=np.int8)
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[os.path.basename(videos_list[video_num])]['length']
m_items_test = m_items.clone()

model.eval()
t1 = time.time()
for k,(imgs) in enumerate(test_batch):

    if k == label_length-4*(video_num+1):
        video_num += 1
        label_length += videos[os.path.basename(videos_list[video_num])]['length']

    imgs = Variable(imgs).to(device, non_blocking=device.type == 'cuda')
    outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
    output_img = outputs.cpu().data.numpy()
    output_img = output_img.squeeze(0)
    mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
    mse_feas = compactness_loss.item()
    
    output_img = np.transpose(output_img, (1, 2, 0))


    point_sc = point_score(outputs, imgs[:,3*4:])

    if  point_sc < args.th:
        query = F.normalize(feas, dim=1)
        query = query.permute(0,2,3,1) # b X h X w X d
        m_items_test = model.memory.update(query, m_items_test, False)
    psnr_dict = {}

    current_video_name = os.path.basename(videos_list[video_num])
    psnr_list[current_video_name].append(psnr(mse_imgs))
    feature_distance_list[current_video_name].append(mse_feas)

t2 = time.time()
anomaly_scores_by_video = []
for video in sorted(videos_list):
    video_name = os.path.basename(video)
    normality_scores = np.asarray(
        score_sum(anomaly_score_list(psnr_list[video_name]),
                  anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha),
        dtype=np.float32)
    anomaly_scores_by_video.append(1.0 - normality_scores)

video_names = [os.path.basename(video) for video in sorted(videos_list)]
score_lengths = np.asarray([len(scores) for scores in anomaly_scores_by_video], dtype=np.int64)
anomaly_score_total_list = np.concatenate(anomaly_scores_by_video)
score_labels = np.concatenate([labels_by_video[name] for name in video_names])

score_output = args.score_output
if score_output is None:
    score_output = os.path.join(os.path.dirname(os.path.abspath(args.model_dir)), 'scores_pred.npz')
score_parent = os.path.dirname(os.path.abspath(score_output))
os.makedirs(score_parent, exist_ok=True)
np.savez(
    score_output,
    dataset=np.asarray(args.dataset_type),
    branch=np.asarray('pred'),
    video_names=np.asarray(video_names),
    lengths=score_lengths,
    scores=anomaly_score_total_list,
    labels=score_labels)

output_dir = os.path.dirname(os.path.abspath(args.model_dir))
plt.plot(anomaly_score_total_list)
plt.savefig(os.path.join(output_dir, 'graph.png'))
plt.close()

accuracy = AUC(anomaly_score_total_list, np.expand_dims(labels_list, 0))

print(t2-t1)
print('AUC: ', accuracy*100, '%')
print('Scores saved to:', score_output)


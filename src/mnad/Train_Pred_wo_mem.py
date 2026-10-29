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
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils_pred import DataLoader
from model.Prediction_wo_memory import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import wandb
wandb.init(project="mnad", mode=os.environ.get("WANDB_MODE", "disabled"))

import argparse


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='anomaly-score weight')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='no_mem', help='directory of log')
parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')

args = parser.parse_args()

torch.manual_seed(2020)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpus)

torch.backends.cudnn.enabled = True
device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device if args.device != 'auto' else 'cpu')
if device.type == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but is unavailable.')
print('Using device:', device)

train_folder = args.dataset_path+'/'+args.dataset_type+"/training/frames"
test_folder = args.dataset_path+'/'+args.dataset_type+"/testing/frames"
print(train_folder)
train_dataset = DataLoader(train_folder, transforms.Compose([
             transforms.ToTensor(),          
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)
print("Train loader done")
test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)
print("test loader done")
train_size = len(train_dataset)
test_size = len(test_dataset)
print("Train Size:" + str(train_size))
print("Test Size:" + str(test_size))
train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)


model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params_count = sum([np.prod(p.size()) for p in model_parameters])
print(params_count)
params_encoder =  list(model.encoder.parameters()) 
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr = args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
model.to(device)
wandb.watch(model)

print(model)
print("CP 0")
log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

loss_func_mse = nn.MSELoss(reduction='none')


m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float, device=device), dim=1)
for epoch in range(args.epochs):
    epoch_start = time.time()
    labels_list = []
    model.train()
    
    start = time.time()
    train_loss = AverageMeter()
    step_0 = time.time()

    for j,(imgs) in enumerate(train_batch):
        imgs = Variable(imgs).to(device, non_blocking=device.type == 'cuda')
        
        outputs = model.forward(imgs[:,0:12], m_items, True)
        
        
        optimizer.zero_grad()
        loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:,12:]))
        loss = loss_pixel
        train_loss.update(loss.item(),imgs.size(0))
        if(j%100 == 0):
          step_100 = time.time()
          print(f"[{j}] Train loss  {train_loss.avg} Time 100 steps: {step_100-step_0}")
          step_0 = step_100
        loss.backward(retain_graph=False)
        wandb.log({"Loss": loss, "Average Loss": train_loss.avg})
        optimizer.step()
    if (epoch%5 == 0):
        torch.save(model, os.path.join(log_dir, 'model_final'+str(epoch)+'.pth'))
    scheduler.step()
    epoch_end = time.time()
    print('----------------------------------------')
    print('Epoch:', epoch+1)
    print('Loss: Reconstruction {:.6f}'.format(loss_pixel.item()))
    print("Epoch time taken:", str(epoch_end-epoch_start))
    print('----------------------------------------')
    
print('Training is finished')
print(log_dir)
torch.save(model, os.path.join(log_dir, 'model.pth'))
    


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
from model.utils import DataLoader
from model.reconstruction_model import *
from sklearn.metrics import roc_auc_score
from utils import *
from kmeans_memory import build_kmeans_memory
import random
from wandb_compat import wandb
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
parser.add_argument('--loss_compact', type=float, default=0.01, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.01, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=2, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='anomaly-score weight')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
parser.add_argument('--kmeans_init', action='store_true', help='replace memory with K-means centers after warm-up')
parser.add_argument('--kmeans_warmup_epochs', type=int, default=5, help='number of normal warm-up epochs')
parser.add_argument('--kmeans_seed', type=int, default=2020, help='random seed used by K-means')

args = parser.parse_args()

if args.kmeans_init and not 1 <= args.kmeans_warmup_epochs < args.epochs:
    parser.error('--kmeans_warmup_epochs must be at least 1 and smaller than --epochs')

torch.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2020)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpus)

torch.backends.cudnn.enabled = True
device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device if args.device != 'auto' else 'cpu')
if device.type == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but is unavailable.')
print('Using device:', device)

train_folder = os.path.join(args.dataset_path, args.dataset_type, "training", "frames")
test_folder = os.path.join(args.dataset_path, args.dataset_type, "testing", "frames")

train_dataset = DataLoader(train_folder, transforms.Compose([
             transforms.ToTensor(),          
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1,num_pred = 0)

test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1,num_pred = 0)

train_size = len(train_dataset)
test_size = len(test_dataset)

train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
kmeans_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers, drop_last=False)
test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)


model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)

params_encoder =  list(model.encoder.parameters()) 
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr = args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
model.to(device)
wandb.watch(model)

log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

loss_func_mse = nn.MSELoss(reduction='none')

best_test = float("inf")
m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float, device=device), dim=1)

for epoch in range(args.epochs):
    labels_list = []
    model.train()
    start = time.time()
    train_loss = AverageMeter()
    for j,(imgs) in enumerate(train_batch):
        
        imgs = Variable(imgs).to(device, non_blocking=device.type == 'cuda')
        outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs, m_items, True)
        optimizer.zero_grad()
        loss_pixel = torch.mean(loss_func_mse(outputs, imgs))
        loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
        train_loss.update(loss.item(),imgs.size(0))
        if(j%100 == 0):
          print(f"[{j}] Train loss  {train_loss.avg}")
        loss.backward(retain_graph=True)
        optimizer.step()
        wandb.log({'Train_loss_avg': train_loss.avg,'Loss': loss, "Compactness Loss": compactness_loss, "Separateness Loss": separateness_loss})
    completed_epochs = epoch + 1
    if completed_epochs % 5 == 0:
        torch.save(model, os.path.join(log_dir, f'{completed_epochs}_model.pth'))
        torch.save(m_items, os.path.join(log_dir, f'{completed_epochs}_keys.pt'))

    if args.kmeans_init and completed_epochs == args.kmeans_warmup_epochs:
        print('Starting K-means memory initialization...')
        m_items, query_count = build_kmeans_memory(
            model=model,
            loader=kmeans_batch,
            device=device,
            memory_size=args.msize,
            input_channels=None,
            seed=args.kmeans_seed)
        torch.save(m_items, os.path.join(log_dir, f'{completed_epochs}_kmeans_keys.pt'))
        print(f'Memory replaced after epoch {completed_epochs}; queries used: {query_count}')
    scheduler.step()
    
    print('----------------------------------------')
    print('Epoch:', epoch+1)
    print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item()))
    print('Memory_items:')
    print(m_items)
    print('----------------------------------------')
    print(f'Train loss avg: {train_loss.avg}')

torch.save(model, os.path.join(log_dir, 'model.pth'))
torch.save(m_items, os.path.join(log_dir, 'keys.pt'))

print('Training is finished')
print(log_dir)



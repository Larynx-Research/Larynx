import argparse
import os
import time
import datetime
from tqdm import trange
import pickle

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
import model

def model_callable():
    kwargs = sorted(name for name in model.__dict__
        if name.islower() and not name.startswith("__")
        and callable(model.__dict__[name]) and name.startswith("_"))
    return kwargs


parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', default="../../data/calib_image_data", type=str,
                    help='path to dataset')

group = parser.add_mutually_exclusive_group()

group.add_argument('--split_value', default=0.8, type=float,
                   help='test-val split proportion between 0 (only test) and 1 (only train), '
                        'will be overwritten if a split file is set')
parser.add_argument('--arch', '-a', metavar='ARCH', default='flownetc',
                    choices=callable,)
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--multiscale-weights', '-w', default=[0.005,0.01,0.02,0.08,0.32], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',default=False,
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--div-flow', default=20,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--milestones', default=[100,150,200], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')


def main():


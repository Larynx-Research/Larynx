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


parser = argparse.ArgumentParser(description='PyTorch WAVENET Training on LIBRE SPEECH DATASET',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', default="E:/data/LJSpeech-1.1", type=str,
                    help='path to dataset')

group = parser.add_mutually_exclusive_group()

group.add_argument('--split_value', default=0.8, type=float,
                   help='test-val split proportion between 0 (only test) and 1 (only train), '
                        'will be overwritten if a split file is set')
parser.add_argument('--arch', '-a', metavar='ARCH', default='wavenet',
                    choices=callable,)
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch_size', default=32, type=int,
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

## ----------------------- global variables ----------------------- ##

## use log liklihood max
NLLLOSS = -1
n_iters = 0
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def main():
    global args, best_cross_entropy
    args = parser.parse_args()
    main_epoch = 0
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    save_path = f'{args.arch}_{args.solver}_{args.epochs}_bs{args.batch_size}_time{timestamp}_lr{args.lr}'
        
    save_path = os.path.join("./pretrained/", save_path)
    print(f"=> will save everything to {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

## --------------------- transforming the data (BUT HOW) --------------------- ##

    input_transform = transforms.Compose([
            transforms.Resize((100)),
            #RandomTranslate(10),
            transforms.ColorJitter(brightness=.3, contrast=0, saturation=0, hue=0),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
            transforms.Normalize(mean=[.45,.432,.411], std=[1,1,1]),
        ])

## --------------------- loading and concatinating the data --------------------- ##

    print(f"=> fetching image pairs from {args.data}")   
    train_set, validation_set = load_dataset(args.data, transforms=None, split = args.split_value)

    print(f"=> {len(validation_set) + len(train_set)} samples found, {len(train_set)} train samples and {len(validation_set)} test samples")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validate_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True)

## --------------------- WAVENET from model.py --------------------- ##

    in_channel = 1
    out_channel = 1
    kernel_size = 1
    stack_size = 16
    model = wavenet(in_channel, out_channel, kernel_size, stack_size).to(device)

    if args.pretrained is not None:
        with open(args.pretrained, 'rb') as pickle_file:
            network_data = pickle.load(pickle_file)

        model.load_state_dict(network_data["state_dict"])
        main_epoch = network_data['epoch']
        print(f"=> creating model {args.arch}")
    else:
        network_data = None
        print(f"=> No pretrained weights ")

## --------------------- Checking and selecting a optimizer [SGD, ADAM] --------------------- ##

    if args.solver not in ['adam', 'sgd']:
        print("=> enter a supported optimizer")
        return 

    print(f'=> settting {args.solver} optimizer')
    param_groups = [{'params': model.parameters(), 'weight_decay': args.bias_decay},
            {'params': model.parameters(), 'weight_decay': args.weight_decay}]

    if device.type == 'cuda':
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(args.momentum, args.beta)) if args.solver == 'adam' else torch.optim.SGD(param_groups, args.lr, momentum=args.momentum)
    
    if args.evaluate:
        best_cross_entropy = validation(validate_loader, model, 0, output_writers, loss_function)
        return

## --------------------- Scheduler and Loss Function --------------------- ##

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=.5)

    loss_function = torch.nn.NLLLoss()

    
## --------------------- Validation Step --------------------- ##
    print("=> training go look tensorboard for more stuff")
    for epoch in (r := trange(args.start_epoch, args.epochs)):

        with torch.no_grad():
            NLL_loss_val, display_val = validation(validate_loader, model, epoch+main_epoch, loss_function)

## --------------------- Training Loop --------------------- ##

        train_loss_NLL, display = train(train_loader, model,
                optimizer, epoch+main_epoch, loss_function)
        
        scheduler.step()
        
        if best_NLL < 0:
            best_neg_loss = NLL_loss_val

        is_best = NLL_loss_val < best_neg_loss
        best_CE = min(NLL_loss_val, best_NLL)

## --------------------- Saving on every epoch --------------------- ##

        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_EPE': best_NLL,
            'div_flow': args.div_flow
        }, is_best, save_path)
        
        r.set_description(f"Epoch: {epoch} ; Train_stuff: {display},, val_Stuff: {display_val}")

## --------------------- TRAIN function for the training loop --------------------- ##

def train(train_loader, model, optimizer, epoch, loss_function):
    global n_iters, args

    epoch_size = len(train_loader)+main_epoch if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    losses = []
    model.train()
    start_time = time.time()

## --------------------- Training --------------------- ##

# wave is the waveform and speech is the text and you have to pad waveform ! search ways to do
    for i, (wave, speech) in enumerate(train_loader):
        wave = wave.to(device)
        speech = speech.to(device)
    
        pred_wave = model(speech)

        loss = loss_function(pred_wave, wave)
        
        losses.append(float(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end = time.time()
    batch_time = end - start_time
        
## --------------------- Stuff to display at output --------------------- ##

       #     display = (' Epoch: [{0}][{1}/{2}] ; Time {3} ; Avg MSELoss {4} ; yaw_MSE {5} ; pitch_MSE {6}').format(epoch, 
       #             i, epoch_size, batch_time, sum(losses)/len(losses), yaw_MSE.item(), pitch_MSE.item())
       #     print(display)
       # n_iters += 1
       # if i >= epoch_size:
       #     break
    display=1
    
    return sum(losses)/len(losses), display

def validation(val_loader, model, epoch, loss_function):
    global args

    model.eval()
    
    start = time.time()
    for i, (wave, speech) in enumerate(val_loader):
        wave = wave.to(device)
        speech = speech.to(device)

        pred_wave = model(speech)

        loss = loss_function(pred_wave, wave)

    end = time.time()
    batch_time = end-start

       # if i < len(output_writers):
       #     if epoch == 0:
       #         mean_values = torch.tensor([0.45,0.432,0.411], dtype=input.dtype).view(3,1,1)
       #         output_writers[i].add_image('GroundTruth', flow2rgb(args.div_flow * target[0], max_value=10), 0)
       #         output_writers[i].add_image('Inputs', (input[0,:3].cpu() + mean_values).clamp(0,1), 0)
       #         output_writers[i].add_image('Inputs', (input[0,3:].cpu() + mean_values).clamp(0,1), 1)
       #     output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)

       # if i % args.print_freq == 0:
       #     display_val = ('Test: [{0}/{1}] ; Loss {2}').format(i, len(val_loader), loss.item())
       #     print(display_val)
       #     print(f"=> Values: Actual yaw: {np.argmax(yaw)} ; Pred yaw: {np.argmax(pred_yaw)} --- Actual pitch : {np.argmax(pitch)} ; Pred pitch : {np.argmax(pred_pitch)}")

    display_val=1
    return loss.item(), display_val
        
if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()


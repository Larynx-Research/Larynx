from __future__ import division
import os, glob
import shutil
import cv2

import random
import numpy as np
import numbers
import types
import argparse

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import pickle


def data_seeker(root):
    input_img = (glob.glob(os.path.join(root)))
    print(np.array(input_img).shape)
    return input_img


def load_dataset(root, transform=None, split=None):
    train_data = data_seeker(root)
    global_buffer = {}
    for i in train_data[:1]:
        waveform, sample_rate = torchaudio.load(i)
        global_buffer[i.split("\\")[1]] = waveform

    print(global_buffer)

def lazy_load_dataset(root, transform, split):
    pass






root = "E:/data/LJSpeech-1.1/wavs/*"


load_dataset(root)



from __future__ import division
import os, glob
import shutil
import cv2

import random
import numpy as np
import numbers
import types
import argparse
import pandas as pd

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import pickle


def data_seeker(root):
    input_img = (glob.glob(os.path.join(root)))
    print(np.array(input_img).shape)
    return input_img

def annote(csv_root, train_data):
    local_buffer = {}
    df = pd.read_csv(csv_root,index)
    for i in train_data:
        local_buffer[i.split("\\")[1]] = df[(i.split("\\")[1]).split(".")[0]]

    print(local_buffer)
    return local_buffer

def load_dataset(root, csv_root, transform=None, split=None):
    train_data = data_seeker(root)
    annotations = annote(csv_root,train_data)


def lazy_load_dataset(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file)
    global_buffer[i.split("\\")[1]] = waveform

    print(global_buffer)
    pass






root = "E:/data/LJSpeech-1.1/wavs/*"
csv_root = "E:/data/LJSpeech-1.1/metadata.csv"


load_dataset(root, csv_root)



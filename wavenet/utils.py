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

CHARSET = " abcdefghijklmnopqrstuvwxyz,.'"

def data_seeker(root):
    input_img = (glob.glob(os.path.join(root,"wavs/*")))
    print(np.array(input_img).shape)
    return input_img

def annote(root, train_data):
    local_buffer = []

    with open(os.path.join(root, "metadata_cp.txt"), encoding="utf8") as txtfile:
        corpus = txtfile.read().split("\n")
        for row in corpus:
            row = row.split("|")
            tokens = [CHARSET.index(c)+1 for c in row[1].lower() if c in CHARSET]
            if len(tokens) <= 250:
                local_buffer.append((os.path.join(root, 'wavs', row[0]+".wav"), tokens))
    print("got metadata", len(local_buffer))
    return local_buffer

def load_dataset(root, transform=None, split=0):
    train_data = data_seeker(root)
    annotations = annote(root,train_data)

    if split:
        train,validate = annotations[0:split], annotations[split:]
        return train, validate
    return annotations, []


def lazy_load_dataset(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file)
    global_buffer[i.split("\\")[1]] = waveform

    print(global_buffer)

root = "E:/data/LJSpeech-1.1/"


load_dataset(root)



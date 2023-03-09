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
global_buffer = {}

def data_seeker(root):
    input_img = (glob.glob(os.path.join(root,"wavs/*")))
    print(np.array(input_img).shape)
    return input_img


def batched_data(data, batch_size, shuffle=True):
    batched_buffer = []
    for i in range(0,len(data),batch_size):
        batched_buffer.append(data[i:batch_size])

    return batched_buffer

class list_dataset(Dataset):
    def __init__(self, root, buffer):
        self.root = root
        self.buffer = buffer

    def __getitem__(self, index):
        wavs = self.buffer[index][0]
        speech = self.buffer[index][1]

        waveform = lazy_load_dataset(wavs)
        return waveform ,speech

    def __len__(self):
        return len(self.buffer)


def annote(root, train_data):
    local_buffer = []

    with open(os.path.join(root, "metadata_cp.txt"), encoding="utf8") as txtfile:
        corpus = txtfile.read().split("\n")
        for row in corpus:
            row = row.split("|")
            row[0] = row[0].split('"')[1] if row[0][0] == '"' else row[0]
            tokens = [CHARSET.index(c)+1 for c in row[1].lower() if c in CHARSET]
            for i in range(250-len(tokens)):
                tokens.append(0)
            if len(tokens) <= 250:
                local_buffer.append([os.path.join(root, 'wavs', row[0]+".wav"), tokens])
    print("got metadata", len(local_buffer))
    return local_buffer

def load_dataset(root, transforms=None, split=0):
    train_data = data_seeker(root)
    annotations = annote(root,train_data)

    if split:
        train_annotations = annotations[0:int(len(annotations)*(split/100))]
        valid_annotations = annotations[int(len(annotations)*(split/100)):]

    train, validate = list_dataset(root, train_annotations), list_dataset(root, valid_annotations)
    return train, validate

def lazy_load_dataset(wav_file):
    if wav_file not in global_buffer:
        waveform, sample_rate = torchaudio.load(wav_file)
        global_buffer[wav_file] = waveform
    return global_buffer[wav_file]

if __name__ == "__main__":

    root = "E:/data/LJSpeech-1.1/"


    load_dataset(root)



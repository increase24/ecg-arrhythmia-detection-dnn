import os
import yaml
import time
import argparse
import numpy as np
from munch import Munch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# custom module
import __init_path
from dataset.ECGDataset_cinc17 import CINC17
from network.ResECG import ResECG
from utils.trainer import validate


def evaluate(params):
    print("Start loading the data....")
    DatasetConfig = Munch(params['DatasetConfig'])
    validset_N = CINC17(os.path.join(DatasetConfig.filelist_root, 'valid_9000ms_N.json'), 256)
    validset_A = CINC17(os.path.join(DatasetConfig.filelist_root, 'valid_9000ms_A.json'), 256)
    print('Finish loading the data....')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ModelConfig = Munch(params['ModelConfig'])
    model = ResECG(ModelConfig)
    model.to(device)
    ckpt_path = './outputs/weights/ResECG_05-29-14-29.pth.tar'
    print("=> loading checkpoint '{}'".format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    # define criterion, optimizer, scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    counter_subplot = 0
    fig = plt.figure(figsize=(12, 8), dpi=300)
    idx2label = {0:'Atrial fibrillation', 1:'Normal rhythm', 2:'Other rhythm', 3:'Noisy recording'}

    plt.subplot(411)
    input, label = validset_N[0]
    ecg_clip = input[0]
    x = np.linspace(0, len(ecg_clip), len(ecg_clip), endpoint=False)
    y = ecg_clip
    plt.plot(x, y, '-')
    plt.ylabel("Amplitude(mV)")
    plt.tight_layout()

    plt.subplot(412)
    input, label = validset_N[1]
    ecg_clip = input[0]
    x = np.linspace(0, len(ecg_clip), len(ecg_clip), endpoint=False)
    y = ecg_clip
    plt.plot(x, y, '-')
    plt.ylabel("Amplitude(mV)")
    plt.tight_layout()

    plt.subplot(413)
    input, label = validset_A[8]
    ecg_clip = input[0]
    x = np.linspace(0, len(ecg_clip), len(ecg_clip), endpoint=False)
    y = ecg_clip
    plt.plot(x, y, '-')
    plt.ylabel("Amplitude(mV)")
    plt.tight_layout()

    plt.subplot(414)
    input, label = validset_N[25]
    ecg_clip = input[0]
    x = np.linspace(0, len(ecg_clip), len(ecg_clip), endpoint=False)
    y = ecg_clip
    plt.plot(x, y, '-')
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude(mV)")
    plt.tight_layout()

    fig.savefig('./figs/predictions.png', dpi=300, bbox_inches='tight')        
                




    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to config file")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    params = yaml.load(open(args.config_file, 'r'))
    evaluate(params)
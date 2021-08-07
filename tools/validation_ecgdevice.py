import os
import yaml
import time
import argparse
import numpy as np
import pandas as pd
from munch import Munch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import signal
b, a = signal.butter(8, 0.2, 'lowpass') 
from scipy.signal import resample
import math
import pywt 
# custom module
import __init_path
from dataset.ECGDataset_cinc17 import CINC17
from network.ResECG import ResECG
from utils.trainer import validate


#封装成函数
def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0
 
def wavelet_noising(new_df):
    data = new_df
    data = data.tolist()  # 将np.ndarray()转为列表
    w = pywt.Wavelet('sym8')
    # [ca3, cd3, cd2, cd1] = pywt.wavedec(data, w, level=3)  # 分解波
    [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)  # 分解波
 
    length1 = len(cd1)
    length0 = len(data)
 
    Cd1 = np.array(cd1)
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)
 
    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0 ), math.e))
    usecoeffs = []
    usecoeffs.append(ca5)  # 向列表末尾添加对象
    #软硬阈值折中的方法
    a = 0.5
 
    for k in range(length1):
        if (abs(cd1[k]) >= lamda):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
        else:
            cd1[k] = 0.0
 
    length2 = len(cd2)
    for k in range(length2):
        if (abs(cd2[k]) >= lamda):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
        else:
            cd2[k] = 0.0
 
    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k]) >= lamda):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
        else:
            cd3[k] = 0.0
 
    length4 = len(cd4)
    for k in range(length4):
        if (abs(cd4[k]) >= lamda):
            cd4[k] = sgn(cd4[k]) * (abs(cd4[k]) - a * lamda)
        else:
            cd4[k] = 0.0
 
    length5 = len(cd5)
    for k in range(length5):
        if (abs(cd5[k]) >= lamda):
            cd5[k] = sgn(cd5[k]) * (abs(cd5[k]) - a * lamda)
        else:
            cd5[k] = 0.0
 
    usecoeffs.append(cd5)
    usecoeffs.append(cd4)
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)
    return recoeffs


def evaluate(params):
    print("Start loading the data....")
    DatasetConfig = Munch(params['DatasetConfig'])
    validset = CINC17(os.path.join(DatasetConfig.filelist_root, DatasetConfig.validlist), 256)
    print('Finish loading the data....')

    fig = plt.figure(figsize=(12, 8), dpi=300)
    plt.subplot(511)
    ecg_signal = pd.read_csv('./test_data/ECGBBBB001.txt', sep = '\t', header=None)
    ecg_array = ecg_signal.iloc[:,0].values/6400.0
    ecg_array = resample(ecg_array, 8960)
    ecg_array = signal.filtfilt(b, a, ecg_array)
    x = np.linspace(0, len(ecg_array), len(ecg_array), endpoint=False)
    y = ecg_array
    plt.plot(x, y, '-')
    plt.ylabel("Amplitude(mV)")
    plt.tight_layout()
    plt.subplot(512)
    ecg_signal = pd.read_csv('./test_data/ECGBBBB002.txt', sep = '\t', header=None)
    ecg_array = ecg_signal.iloc[:,0].values/6400.0
    ecg_array = resample(ecg_array, 8960)
    ecg_array = signal.filtfilt(b, a, ecg_array)
    x = np.linspace(0, len(ecg_array), len(ecg_array), endpoint=False)
    y = ecg_array
    plt.plot(x, y, '-')
    plt.ylabel("Amplitude(mV)")
    plt.tight_layout()
    plt.subplot(513)
    ecg_clip, label = validset[0]
    x = np.linspace(0, len(ecg_clip[0]), len(ecg_clip[0]), endpoint=False)
    y = ecg_clip[0]/2
    y = resample(y[:len(y)//2], 8960) 
    plt.plot(x, y, '-')
    plt.ylabel("Amplitude(mV)")
    plt.yticks([-0.5, 0, 0.5, 1.0])
    plt.tight_layout()
    plt.subplot(514)
    ecg_signal = pd.read_csv('./test_data/ECGBBBB003.txt', sep = '\t', header=None)
    ecg_array = ecg_signal.iloc[:,0].values/6400.0
    ecg_array = resample(ecg_array, 8960)
    ecg_array = signal.filtfilt(b, a, ecg_array)
    x = np.linspace(0, len(ecg_array), len(ecg_array), endpoint=False)
    y = ecg_array
    plt.plot(x, y, '-')
    plt.ylabel("Amplitude(mV)")
    plt.tight_layout()
    plt.subplot(515)
    ecg_signal = pd.read_csv('./test_data/ECGBBBB004.txt', sep = '\t', header=None)
    ecg_array = ecg_signal.iloc[:,0].values/6400.0
    ecg_array = resample(ecg_array, 8960)
    #ecg_array = signal.filtfilt(b, a, ecg_array)
    ecg_array =  wavelet_noising(ecg_array)
    x = np.linspace(0, len(ecg_array), len(ecg_array), endpoint=False)
    y = ecg_array
    plt.plot(x, y, '-')
    plt.ylabel("Amplitude(mV)")
    plt.tight_layout()

    fig.savefig('./figs/validation.png', dpi=300, bbox_inches='tight')        
                


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to config file")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    params = yaml.load(open(args.config_file, 'r'))
    evaluate(params)
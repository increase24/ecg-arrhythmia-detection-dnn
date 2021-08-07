import os
import yaml
import time
import argparse
from munch import Munch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# custom module
import __init_path
from dataset.ECGDataset_cinc17 import CINC17
from network.ResECG import ResECG
from utils.trainer import validate

def evaluate(params):
    print("Start loading the data....")
    DatasetConfig = Munch(params['DatasetConfig'])
    validset = CINC17(os.path.join(DatasetConfig.filelist_root, DatasetConfig.validlist), 256)
    validloader = DataLoader(validset, batch_size = DatasetConfig.batch_size, shuffle=DatasetConfig.shuffle, num_workers=DatasetConfig.num_workers)
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
    valid_loss, valid_acc = validate(validloader, model, device, criterion, 100, True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to config file")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    params = yaml.load(open(args.config_file, 'r'))
    evaluate(params)
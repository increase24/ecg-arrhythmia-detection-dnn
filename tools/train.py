import os
import yaml
import time
import argparse
from munch import Munch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from kornia.losses import FocalLoss
# custom module
import __init_path
from dataset.ECGDataset_cinc17 import CINC17
from network.ResECG import ResECG
from utils.trainer import train_epoch, validate
from utils.saver import save_checkpoint

def train(params):
    curr_time = time.strftime("%m-%d-%H-%M", time.localtime())
    print("Start loading the data....")
    DatasetConfig = Munch(params['DatasetConfig'])
    trainset = CINC17(os.path.join(DatasetConfig.filelist_root, DatasetConfig.trainlist), 256)
    validset = CINC17(os.path.join(DatasetConfig.filelist_root, DatasetConfig.validlist), 256)
    trainloader = DataLoader(trainset, batch_size = DatasetConfig.batch_size, shuffle=DatasetConfig.shuffle, num_workers=DatasetConfig.num_workers)
    validloader = DataLoader(validset, batch_size = DatasetConfig.batch_size, shuffle=DatasetConfig.shuffle, num_workers=DatasetConfig.num_workers)
    print('Finish loading the data....')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ModelConfig = Munch(params['ModelConfig'])
    model = ResECG(ModelConfig)
    model.to(device)
    # define criterion, optimizer, scheduler
    OptimizerConfig = Munch(params['OptimizerConfig'])
    if(OptimizerConfig.loss == 'CrossEntropyLoss'):
        if(OptimizerConfig.use_unbalance_weight):
            class_weight = torch.tensor(OptimizerConfig.class_weight)
            class_weight = class_weight*0.2+0.25
        else:
            class_weight = torch.tensor([1.0, 1.0, 1.0, 1.0])
        print('CrossEntropy loss weight:\n', class_weight)
        criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
        print(" > Using CrossEntropy Loss...")
    elif(OptimizerConfig.loss == 'FocalLoss'):
        kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        criterion = FocalLoss(**kwargs).to(device)
        print(" > Using Focal Loss...")
    optimizer = torch.optim.Adam(model.parameters(), lr=OptimizerConfig.learning_rate, amsgrad =True)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30,40], gamma=0.25)
    num_epoches = OptimizerConfig.epoches
    print(" > Training is getting started...")
    print(" > Training takes {} epochs.".format(num_epoches))
    best_acc = 0.0
    for epoch in range(num_epoches):
        # train one epoch
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(trainloader, model, device, criterion, 
            optimizer, epoch, 100) 
        valid_loss, valid_acc = validate(validloader, model, device, criterion, 100, False)
        epoch_end_time = time.time()
        print("epoch cost time: %.4f min" %((epoch_end_time - epoch_start_time)/60))
        #scheduler.step()

        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        print(f'current best_acc: {best_acc}')
        if(is_best):
            print(f'the best accuracy increases to {best_acc}')
            save_checkpoint({
                'epoch': epoch,
                'arch': ModelConfig['model_name'],
                'state_dict': model.state_dict(),
                'best_acc': best_acc
            }, './outputs/weights/', ModelConfig['model_name']+f'_{curr_time}'+'.pth.tar')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to config file")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    params = yaml.load(open(args.config_file, 'r'))
    train(params)
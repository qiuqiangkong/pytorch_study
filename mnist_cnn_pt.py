"""
Summary:  mnist cnn pytorch example. 
          te_err around 1%. 
Author:   Qiuqiang Kong
Usage:    $ python mnist_cnn_pt.py train --init_type=glorot_uniform --optimizer=adam --loss=softmax --lr=1e-3
          $ python mnist_cnn_pt.py train --init_type=glorot_uniform --optimizer=adam --loss=softmax --lr=1e-4 --resume_model_path="models/md_3000iters.tar"
Created:  2017.12.11
Modified: - 
"""
import numpy as np
import time
import argparse
from sklearn import preprocessing
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import prepare_data as pp_data
from data_generator import DataGenerator
from mnist_dnn_pt import train, eval

def uniform_weights(m):
    classname = m.__class__.__name__    
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
        scale = 0.1
        m.weight.data = torch.nn.init.uniform(m.weight.data, -scale, scale)
        m.bias.data.fill_(0.)

def glorot_uniform_weights(m):
    classname = m.__class__.__name__    
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
        # w = torch.nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
        w = torch.nn.init.xavier_uniform(m.weight.data)
        m.weight.data = w
        m.bias.data.fill_(0.)  

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.fc1 = nn.Linear(64*7*7, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        
        return x
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--use_cuda', action='store_true', default=True)
    parser_train.add_argument('--init_type', default='glorot_uniform', choices=['uniform', 'glorot_uniform'])
    parser_train.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'])
    parser_train.add_argument('--loss', default='softmax', choices=['softmax', 'sigmoid'])
    parser_train.add_argument('--lr', type=float, default=1e-3)
    parser_train.add_argument('--resume_model_path', type=str, default="")
    
    args = parser.parse_args()
    if args.mode == "train":
        print(args)
        if args.init_type == 'uniform':
            init_weights = uniform_weights
        elif args.init_type == 'glorot_uniform':
            init_weights = glorot_uniform_weights
        else:
            raise Exception("Incorrect init_type!")
        args.loss = 'softmax'
        model = CNN()
        train(model, init_weights=init_weights, args=args)
    else:
        raise Exception("Error!")
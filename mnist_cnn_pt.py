"""
Summary:  mnist cnn pytorch example. 
Author:   Qiuqiang Kong
Usage:    $ python mnist_cnn_pt.py train --init_type=glorot_uniform --optimizer=adam --loss=softmax
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
from mnist_dnn_pt import train, eval, uniform_weights, glorot_uniform_weights

def back_hook1(grad):
    print 'back_hook1'
    print grad

class CNN(nn.Module):
    def __init__(self, loss_type):
        super(CNN, self).__init__()
        
        self.loss_type = loss_type
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
        x5 = self.fc2(x)
        
        if self.loss_type == 'softmax':
            return F.log_softmax(x5)
        elif self.loss_type == 'sigmoid':
            return F.sigmoid(x5)
        else:
            raise Exception("Incorrect loss_type!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_a = subparsers.add_parser('train')
    parser_a.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    parser_a.add_argument('--init_type', default='glorot_uniform', 
                          choices=['uniform', 'glorot_uniform'])
    parser_a.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'])
    parser_a.add_argument('--loss', default='softmax', choices=['softmax', 'sigmoid'])
    
    args = parser.parse_args()
    if args.mode == "train":
        print(args)
        train(builder=CNN, args=args)
    else:
        raise Exception("Error!")
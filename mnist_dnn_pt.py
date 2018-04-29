"""
Summary:  mnist dnn pytorch example. 
          te_err around 2%. 
Author:   Qiuqiang Kong
Usage:    $ CUDA_VISIBLE_DEVICES=1 python mnist_dnn_pt.py train --optimizer=adam --output_type=softmax --lr=1e-3
          $ CUDA_VISIBLE_DEVICES=1 python mnist_dnn_pt.py train --optimizer=adam --output_type=softmax --lr=1e-4 --resume_model_path="models/md_1000_iters.tar"
          $ CUDA_VISIBLE_DEVICES=1 python mnist_dnn_pt.py inference --iteration=1000 --output_type=softmax
Created:  2017.12.09
Modified: 2017.12.12
          2018.04.08
"""
import numpy as np
import time
import os
import argparse
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

import prepare_data as pp_data
from data_generator import DataGenerator


def back_hook1(grad):
    print('back_hook1')
    print(grad)


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()
    
    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)
    
    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.weight.data.fill_(1.)


class DNN(nn.Module):
    def __init__(self, output_type):
        super(DNN, self).__init__()
        
        n_hid = 500
        self.output_type = output_type
        self.fc1 = nn.Linear(784, n_hid, bias=True)
        self.fc2 = nn.Linear(n_hid, n_hid, bias=True)
        self.fc3 = nn.Linear(n_hid, n_hid, bias=True)
        self.fc4 = nn.Linear(n_hid, 10, bias=True)
        
        self.bn1 = nn.BatchNorm2d(n_hid)
        self.bn2 = nn.BatchNorm2d(n_hid)
        self.bn3 = nn.BatchNorm2d(n_hid)
    
        self.init_weights()
    
    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        
    def forward(self, x):
        drop_p = 0.2
        x2 = F.dropout(F.relu(self.bn1(self.fc1(x))), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.bn2(self.fc2(x2))), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.bn3(self.fc3(x3))), p=drop_p, training=self.training)
        x5 = self.fc4(x4)
 
        if False:
            x3.register_hook(back_hook1)      # observe backward gradient
            print(x3)          # observe forward
        
        if self.output_type == 'softmax':
            return F.log_softmax(x5, dim=-1)   # Return linear here and use F.nll_loss() later. 
        elif self.output_type == 'sigmoid':
            return F.sigmoid(x5)    # Use F.binary_cross_entropy() later. 
        else:
            raise Exception("Incorrect output_type!")
            
    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False
    
    
def move_x_to_gpu(x, cuda, volatile=False):
    x = torch.Tensor(x)
    if cuda:
        x = x.cuda()
    x = Variable(x, volatile=volatile)
    return x
    

def move_y_to_gpu(y, output_type, cuda, volatile=False):
    if output_type == 'softmax':
        y = torch.LongTensor(y)
    elif output_type == 'sigmoid':
        y = torch.Tensor(y)
        
    if cuda:
        y = y.cuda()
    y = Variable(y, volatile=volatile)
    return y
    
    
def calculate_error(output, target):
    error = torch.sum((output != target).float()) / target.shape[0]
    return error
    
    
def evaluate(model, output_type, gen, xs, ys, cuda):
    model.eval()
    output_all = []
    target_all = []
    
    iteration = 0
    max_iteration = -1
    
    # Evaluate in mini-batch
    for (batch_x, batch_y) in gen.generate(xs=xs, ys=ys):
        
        if iteration == max_iteration:
            break
        
        batch_x = move_x_to_gpu(batch_x, cuda, volatile=True)
        batch_y = move_y_to_gpu(batch_y, output_type, cuda, volatile=True)
        
        batch_output = model(batch_x)
        
        (_, batch_output) = torch.max(batch_output, dim=-1)
        
        output_all.append(batch_output)
        target_all.append(batch_y)
        
        iteration += 1

    output_all = torch.cat(output_all, dim=0)
    target_all = torch.cat(target_all, dim=0)
    
    if output_type == 'sigmoid':
        (_, target_all) = torch.max(target_all, dim=-1)
        
    error = calculate_error(output_all, target_all)
    error = error.data.cpu().numpy()[0]
        
    return error
          
    
def train(args):
    cuda = args.use_cuda and torch.cuda.is_available()
    opt_type = args.optimizer
    output_type = args.output_type
    lr = args.lr
    freeze = args.freeze
    resume_model_path = args.resume_model_path
    print("cuda:", cuda)
    
    # Load data
    (tr_x, tr_y, va_x, va_y, te_x, te_y) = pp_data.load_data()

    n_out = 10
    
    if output_type == 'sigmoid':
        tr_y = pp_data.sparse_to_categorical(tr_y, n_out)
        te_y = pp_data.sparse_to_categorical(te_y, n_out)
        
    print("tr_x.shape:", tr_x.shape)
    
    tr_x = tr_x.astype(np.float32)
    va_x = va_x.astype(np.float32)
    te_x = te_x.astype(np.float32)
    tr_y = tr_y.astype(np.int64)
    va_y = va_y.astype(np.int64)
    te_y = te_y.astype(np.int64)
    
    # Scale
    scaler = preprocessing.StandardScaler().fit(tr_x)
    tr_x = scaler.transform(tr_x)
    va_x = scaler.transform(va_x)
    te_x = scaler.transform(te_x)
    print(tr_x.dtype, tr_y.dtype)
    
    # Model
    model = DNN(output_type)
    
    if os.path.isfile(resume_model_path):
        # Load weights
        print("Loading checkpoint {}".format(resume_model_path))
        checkpoint = torch.load(resume_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        iteration = checkpoint['iteration']
    else:
        # Randomly init weights
        print("Train from random initialization. ")
        iteration = 0
        
    # Move model to GPU
    if cuda:
        model.cuda()

    # Freeze parameters
    if freeze:    
        model.freeze_layer(model.fc1)
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        params = model.parameters()

    # Data generator
    batch_size = 500
    print("{:.2f} iterations / epoch".format(tr_x.shape[0] / float(batch_size)))
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=20)
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test')
    
    
    # Optimizer
    if opt_type == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=0.9)
    elif opt_type == 'adam':
        optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)
    else:
        raise Exception("Optimizer wrong!")
    
    # Train
    bgn_train_time = time.time()
    for batch_x, batch_y in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        
        # Evaluate
        if iteration % 100 == 0:
            fin_train_time = time.time()
            eval_time = time.time()
            tr_error = evaluate(model, output_type, eval_tr_gen, [tr_x], [tr_y], cuda)
            te_error = evaluate(model, output_type, eval_te_gen, [te_x], [te_y], cuda)
            print("Iteration: {}, train err: {}, test err: {}, train time: {}, eval time: {}".format(
                  iteration, tr_error, te_error, fin_train_time - bgn_train_time, time.time() - eval_time))
            bgn_train_time = time.time()

        # Move data to GPU
        load_time = time.time()
        batch_x = move_x_to_gpu(batch_x, cuda)
        batch_y = move_y_to_gpu(batch_y, output_type, cuda)
        
        if False:
            print("Load data time: {}".format(time.time() - load_time))
        
        # Print wights
        if False:
            print(weights)
            print(model.fc1)
            print(model.fc1.weight)
        
        # Forward
        forward_time = time.time()
        model.train()
        output = model(batch_x)
        
        # Loss
        if output_type == 'softmax':
            loss = F.nll_loss(output, batch_y)
        elif output_type == 'sigmoid':
            loss = F.binary_cross_entropy(output, batch_y)
        else:
            raise Exception("Incorrect output_type!")
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if False:
            print(model.fc2.weight.data)
            print(model.fc2.weight.grad.data)
            pause

        if False:
            print("Train time: {}".format(time.time() - forward_time))
        
        iteration += 1
        
        # Save model
        if iteration % 1000 == 0 and iteration > 0:
            save_out_dict = {'iteration': iteration, 
                             'state_dict': model.state_dict(), 
                             'optimizer': optimizer.state_dict(), 
                             'te_error': te_error, }
            save_out_path = "models/md_{}_iters.tar".format(iteration)
            pp_data.create_folder(os.path.dirname(save_out_path))
            torch.save(save_out_dict, save_out_path)
            print("Save model to {}".format(save_out_path))
        

def forward_in_batch(model, x, batch_size, cuda):
    model.eval()
    batch_num = int(np.ceil(len(x) / float(batch_size)))
    output_all = []
    
    for i1 in range(batch_num):
        batch_x = x[i1 * batch_size : (i1 + 1) * batch_size]
        batch_x = move_x_to_gpu(batch_x, cuda, volatile=True)
        output = model(batch_x)
        output_all.append(output)
    
    output_all = torch.cat(output_all, dim=0)
    return output_all
            
            
def inference(args):
    output_type = args.output_type
    iteration = args.iteration
    cuda = args.use_cuda and torch.cuda.is_available()
    
    # Load data
    (tr_x, tr_y, va_x, va_y, te_x, te_y) = pp_data.load_data()
    tr_x = tr_x.astype(np.float32)
    va_x = va_x.astype(np.float32)
    te_x = te_x.astype(np.float32)
    tr_y = tr_y.astype(np.int64)
    va_y = va_y.astype(np.int64)
    te_y = te_y.astype(np.int64)

    # Scale
    scaler = preprocessing.StandardScaler().fit(tr_x)
    tr_x = scaler.transform(tr_x)
    te_x = scaler.transform(te_x)

    if output_type == 'sigmoid':
        n_out = 10
        te_y = pp_data.sparse_to_categorical(te_y, n_out)

    # Load model
    model = DNN(output_type)
    checkpoint = torch.load(os.path.join("models", "md_{}_iters.tar".format(iteration)))
    model.load_state_dict(checkpoint['state_dict'])
    
    if cuda:
        model.cuda()

    # Inference
    batch_size = 128
    output = forward_in_batch(model, te_x, batch_size, cuda)
    (_, output) = torch.max(output, dim=-1)
    
    # Calculate error
    target = move_y_to_gpu(te_y, output_type, cuda)
    
    if output_type == 'sigmoid':
        (_, target) = torch.max(target, dim=-1)

    error = calculate_error(output, target)
    error = error.data.cpu().numpy()[0]
    print("Error:", error)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--use_cuda', action='store_true', default=True)
    parser_train.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'])
    parser_train.add_argument('--output_type', default='softmax', choices=['softmax', 'sigmoid'])
    parser_train.add_argument('--lr', type=float, default=1e-3)
    parser_train.add_argument('--freeze', action='store_true', default=False)
    parser_train.add_argument('--resume_model_path', type=str, default="")
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--use_cuda', action='store_true', default=True)
    parser_inference.add_argument('--output_type', default='softmax', choices=['softmax', 'sigmoid'])
    parser_inference.add_argument('--iteration', type=str, default="")
    
    args = parser.parse_args()
    print(args)
    
    if args.mode == 'train':        
        train(args)
        
    elif args.mode == 'inference':
        inference(args)
        
    else:
        raise Exception('Error!')
        
        
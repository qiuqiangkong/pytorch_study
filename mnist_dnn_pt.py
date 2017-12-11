"""
Summary:  mnist dnn pytorch example. 
Author:   Qiuqiang Kong
Usage:    $ python mnist_dnn_pt.py train --init_type=glorot_uniform --optimizer=adam --loss=softmax
Created:  2017.12.09
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

def back_hook1(grad):
    print 'back_hook1'
    print grad

class DNN(nn.Module):
    def __init__(self, loss_type):
        super(DNN, self).__init__()
        
        self.loss_type = loss_type
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 10)
        
    def forward(self, x):
        drop_p = 0.2
        x2 = F.dropout(F.relu(self.fc1(x)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = self.fc4(x4)
 
        if False:
            x3.register_hook(back_hook1)      # observe backward gradient
            print x3          # observe forward
        
        if self.loss_type == 'softmax':
            return F.log_softmax(x5)
        elif self.loss_type == 'sigmoid':
            return F.sigmoid(x5)
        else:
            raise Exception("Incorrect loss_type!")
    
def eval(model, gen, xs, ys, cuda):
    model.eval()
    pred_all = []
    y_all = []
    for (batch_x, batch_y) in gen.generate(xs=xs, ys=ys):
        batch_x = torch.Tensor(batch_x)
        batch_x = Variable(batch_x, volatile=True)
        if cuda:
            batch_x = batch_x.cuda()
        pred = model(batch_x)
        pred = pred.data.cpu().numpy()
        pred_all.append(pred)
        y_all.append(batch_y)
        
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    err = pp_data.categorical_error(pred_all, y_all)
    return err
    
def uniform_weights(m):
    classname = m.__class__.__name__    
    if classname.find('Linear') != -1:
        scale = 0.1
        m.weight.data = torch.nn.init.uniform(m.weight.data, -scale, scale)
        m.bias.data.fill_(0.)

def glorot_uniform_weights(m):
    classname = m.__class__.__name__    
    if classname.find('Linear') != -1:
        # w = torch.nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
        w = torch.nn.init.xavier_uniform(m.weight.data)
        m.weight.data = w
        m.bias.data.fill_(0.)        
    
def train(builder, args):
    cuda = not args.no_cuda and torch.cuda.is_available()
    init_type = args.init_type
    opt_type = args.optimizer
    loss_type = args.loss
    
    # Load data. 
    (tr_x, tr_y, va_x, va_y, te_x, te_y) = pp_data.load_data()
    n_out = 10
    tr_y = pp_data.sparse_to_categorical(tr_y, n_out)
    te_y = pp_data.sparse_to_categorical(te_y, n_out)
    print("tr_x.shape:", tr_x.shape)
    
    # Scale
    scaler = preprocessing.StandardScaler().fit(tr_x)
    tr_x = scaler.transform(tr_x)
    va_x = scaler.transform(va_x)
    te_x = scaler.transform(te_x)
    
    # Model & init weights. 
    model = builder(loss_type)
    if init_type == 'uniform':
        model.apply(uniform_weights)
    elif init_type == 'glorot_uniform':
        model.apply(glorot_uniform_weights)
    else:
        raise Exception("Incorrect init_type!")
        
    # Move model to GPU. 
    if cuda:
        model.cuda()

    # Data generator. 
    batch_size = 500
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=20)
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test')
    
    # Optimizer. 
    if opt_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    elif opt_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        raise Exception("Optimizer wrong!")
    
    # Evaluate. 
    tr_err = eval(model, eval_tr_gen, [tr_x], [tr_y], cuda)
    print("Initial train err: %f" % tr_err)
    te_err = eval(model, eval_te_gen, [te_x], [te_y], cuda)
    print("Initial test: %f" % te_err)
    
    # Train. 
    loss_func = nn.CrossEntropyLoss()
    iter = 0
    for batch_x, batch_y in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        # Move data to GPU. 
        t1 = time.time()
        batch_x = torch.Tensor(batch_x)
        batch_y = torch.LongTensor(np.argmax(batch_y, axis=-1))
        batch_y = torch.LongTensor(batch_y)
        batch_x = Variable(batch_x)
        batch_y = Variable(batch_y)
        if cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        
        if False:
            print("Load data time:", time.time() - t1)
        
        # Print wights. 
        if False:
            print weights
            print model.fc1
            print model.fc1.weight
        
        # Forward. 
        t1 = time.time()
        model.train()
        output = model(batch_x)
        
        # Loss. 
        if loss_type == 'softmax':
            loss = F.nll_loss(output, batch_y)
        elif loss_type == 'sigmoid':
            binary_cross_entropy(output, batch_y)
        else:
            raise Exception("Incorrect loss_type!")
            
        # Backward. 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if False:
            print("Train time:", time.time() - t1)
        
        iter += 1
        
        # Evaluate. 
        loss_ary = []
        if iter % 100 == 0:
            if False:
                t1 = time.time()
                tr_err = eval(model, eval_tr_gen, [tr_x], [tr_y], cuda)
                print("Iter: %d, train err: %f, time: %s" % (iter, tr_err, time.time() - t1))
            t1 = time.time()
            te_err = eval(model, eval_te_gen, [te_x], [te_y], cuda)
            print("Iter: %d, test err: %f, time: %s" % (iter, te_err, time.time() - t1))
        
        
    
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
        train(builder=DNN, args=args)
    else:
        raise Exception("Error!")
"""
Summary:  mnist dnn pytorch example. 
          te_err around 2%. 
Author:   Qiuqiang Kong
Usage:    $ CUDA_VISIBLE_DEVICES=1 python test9.py train --init_type=glorot_uniform --optimizer=adam --loss=softmax --lr=1e-3
          $ python mnist_dnn_pt.py train --init_type=glorot_uniform --optimizer=adam --loss=softmax --lr=1e-4 --resume_model_path="models/md_3000iters.tar"
Created:  2017.12.09
Modified: 2017.12.12
"""
import numpy as np
import time
import os
import argparse
from sklearn import preprocessing
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

import prepare_data as pp_data
from data_generator import DataGenerator

def back_hook1(grad):
    print 'back_hook1'
    print grad

class MNIST(data.Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.LongTensor(y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return len(self.x)

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
            return x5   # Return linear here and use F.cross_entropy() later. 
        elif self.loss_type == 'sigmoid':
            return F.sigmoid(x5)    # Use F.binary_cross_entropy() later. 
        else:
            raise Exception("Incorrect loss_type!")
    
def eval(model, data_loader, cuda):
    model.eval()
    pred_all = []
    y_all = []
    for (batch_x, batch_y) in data_loader:
        batch_x = Variable(batch_x, volatile=True)
        if cuda:
            batch_x = batch_x.cuda(async=True)
        pred = model(batch_x)
        pred = pred.data.cpu().numpy()
        pred_all.append(pred)
        y_all.append(batch_y)
        
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    err = pp_data.categorical_error(pred_all, y_all)
    return err
    
def uniform_weights(m):
    if isinstance(m, nn.Linear):
        scale = 0.1
        m.weight.data = torch.nn.init.uniform(m.weight.data, -scale, scale)
        m.bias.data.fill_(0.)

def glorot_uniform_weights(m):
    if isinstance(m, nn.Linear):
        w = torch.nn.init.xavier_uniform(m.weight.data)
        m.weight.data = w
        m.bias.data.fill_(0.)        
    
def train(model, init_weights, args):
    cuda = args.use_cuda and torch.cuda.is_available()
    init_type = args.init_type
    opt_type = args.optimizer
    loss_type = args.loss
    lr = args.lr
    resume_model_path = args.resume_model_path
    print("cuda:", cuda)
    
    # Load data. 
    (tr_x, tr_y, va_x, va_y, te_x, te_y) = pp_data.load_data()
    n_out = 10
    tr_y = pp_data.sparse_to_categorical(tr_y, n_out)
    te_y = pp_data.sparse_to_categorical(te_y, n_out)
    print("tr_x.shape:", tr_x.shape)

    # Scale. 
    scaler = preprocessing.StandardScaler().fit(tr_x)
    tr_x = scaler.transform(tr_x)
    va_x = scaler.transform(va_x)
    te_x = scaler.transform(te_x)
    
    # Dataset. 
    tr_set = MNIST(tr_x, tr_y)
    te_set = MNIST(te_x, te_y)
    batch_size = 500
    tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    eval_tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    eval_te_loader = torch.utils.data.DataLoader(te_set, batch_size=batch_size, shuffle=True, num_workers=2)
    print("%.2f iterations / epoch" % (tr_x.shape[0] / float(batch_size)))
    
    if os.path.isfile(resume_model_path):
        # Load weights. 
        print("Loading checkpoint '%s'" % resume_model_path)
        checkpoint = torch.load(resume_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        iter = checkpoint['iter']
    else:
        # Randomly init weights. 
        print("Train from random initialization. ")
        model.apply(init_weights)
        iter = 0
        
    # Move model to GPU. 
    if cuda:
        model.cuda()

    # Optimizer. 
    if opt_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        raise Exception("Optimizer wrong!")
    
    # Evaluate. 
    t_eval = time.time()
    tr_err = eval(model, eval_tr_loader, cuda)
    te_err = eval(model, eval_te_loader, cuda)
    print("Iter: %d, train err: %f, test err: %f, eval time: %s" % \
          (iter, tr_err, te_err, time.time() - t_eval))
    
    # Train. 
    t1 = time.time()
    t_train = time.time()
    while(True):
        for (batch_x, batch_y) in tr_loader:
            # Move data to GPU. 
            # print(time.time() - t1)
            (_, batch_y) = torch.max(batch_y, -1)

            if cuda:
                batch_x = batch_x.cuda(async=True)
                batch_y = batch_y.cuda(async=True)
            batch_x = Variable(batch_x)
            batch_y = Variable(batch_y)
            
            if False:
                print("Load data time:", time.time() - t1)
            
            # Print wights. 
            if False:
                print weights
                print model.fc1
                print model.fc1.weight
            
            # Forward. 
            while(1):
                t2 = time.time()
                model.train()
                output = model(batch_x)
                
                # Loss. 
                if loss_type == 'softmax':
                    loss = F.cross_entropy(output, batch_y)
                elif loss_type == 'sigmoid':
                    F.binary_cross_entropy(output, batch_y)
                else:
                    raise Exception("Incorrect loss_type!")
                    
                # Backward. 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Set to False to debug print training time per iteration. 
                if True:
                    break
                else:
                    print("Train time:", time.time() - t2)
            
            iter += 1
            
            # Evaluate. 
            loss_ary = []
            if iter % 100 == 0:
                t_eval = time.time()
                tr_err = eval(model, eval_tr_loader, cuda)
                te_err = eval(model, eval_te_loader, cuda)
                print("Iter: %d, train err: %f, test err: %f, train time: %s, eval time: %s" % \
                      (iter, tr_err, te_err, time.time() - t_train, time.time() - t_eval))
                t_train = time.time()
            
            # Save model. 
            if iter % 1000 == 0:
                save_out_dict = {'iter': iter, 
                                'state_dict': model.state_dict(), 
                                'optimizer': optimizer.state_dict(), 
                                'te_err': te_err, }
                save_out_path = "models/md_%diters.tar" % iter
                pp_data.create_folder(os.path.dirname(save_out_path))
                torch.save(save_out_dict, save_out_path)
                print("Save model to %s" % save_out_path)
    
            t1 = time.time()
    
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
        model = DNN(args.loss)
        train(model=model, init_weights=init_weights, args=args)
    else:
        raise Exception("Error!")
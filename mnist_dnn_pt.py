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

from utilities import load_data, sparse_to_categorical, create_folder, calculate_accuracy
from data_generator import DataGenerator, TestDataGenerator


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
        
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.bn3 = nn.BatchNorm1d(n_hid)
    
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
            raise Exception('Incorrect output_type!')
            
    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False
    
    
def move_data_to_gpu(x, cuda):

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception('Error!')

    if cuda:
        x = x.cuda()

    return x
    
    
def calculate_error(output, target):
    error = torch.sum((output != target).float()) / target.shape[0]
    return error
    
    
def evaluate(model, generator, data_type, max_iteration, cuda):
    """Evaluate
    
    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      max_iteration: int, maximum iteration for validation
      cuda: bool.
      
    Returns:
      accuracy: float
    """
    
    # Generate function
    generate_func = generator.generate_validate(data_type=data_type, 
                                                shuffle=True, 
                                                max_iteration=max_iteration)
            
    # Forward
    dict = forward(model=model, 
                   generate_func=generate_func, 
                   cuda=cuda, 
                   return_target=True)

    outputs = dict['output']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)

    predictions = np.argmax(outputs, axis=-1)   # (audios_num,)

    # Evaluate
    classes_num = outputs.shape[-1]
    
    loss = F.nll_loss(torch.Tensor(outputs), torch.LongTensor(targets)).numpy()
    loss = float(loss)
    
    accuracy = calculate_accuracy(targets, predictions)

    return accuracy, loss


def forward(model, generate_func, cuda, return_target):
    """Forward data to a model.
    
    Args:
      generate_func: generate function
      cuda: bool
      return_target: bool
      
    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """
    
    outputs = []
    audio_names = []
    
    if return_target:
        targets = []
    
    # Evaluate on mini-batch
    for data in generate_func:
            
        if return_target:
            (batch_x, batch_y, batch_audio_names) = data
            
        else:
            (batch_x, batch_audio_names) = data
            
        batch_x = move_data_to_gpu(batch_x, cuda)
        
        # Predict
        model.eval()
        batch_output = model(batch_x)

        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        audio_names.append(batch_audio_names)
        
        if return_target:
            targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs
    
    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names
    
    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets
        
    return dict
    
    
def train(args):

    # Arguments & parameters
    workspace = args.workspace
    opt_type = args.optimizer
    output_type = args.output_type
    lr = args.lr
    freeze = args.freeze
    resume_model_path = args.resume_model_path
    cuda = args.cuda
    print('cuda:', cuda)
    
    n_out = 10

    # Paths
    models_dir = os.path.join(workspace, 'models')
    create_folder(models_dir)
    
    # Model
    model = DNN(output_type)
    
    if os.path.isfile(resume_model_path):
        # Load weights
        print('Loading checkpoint {}'.format(resume_model_path))
        checkpoint = torch.load(resume_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        iteration = checkpoint['iteration']
    else:
        # Randomly init weights
        print('Train from random initialization. ')
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
    generator = DataGenerator(batch_size=batch_size)
    
    # Optimizer
    if opt_type == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=0.9)
    elif opt_type == 'adam':
        optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)
    else:
        raise Exception('Optimizer wrong!')
    
    # Train
    bgn_train_time = time.time()
    for iteration, (batch_x, batch_y) in enumerate(generator.generate_train()):
                
        # Evaluate
        if iteration % 100 == 0:
            fin_train_time = time.time()
            eval_time = time.time()
            (tr_acc, tr_loss) = evaluate(model, generator, data_type='train', max_iteration=None, cuda=cuda)
            (va_acc, va_loss) = evaluate(model, generator, data_type='validate', max_iteration=None, cuda=cuda)
            print('Iteration: {}, train acc: {}, test acc: {}, train time: {}, eval time: {}'.format(
                  iteration, tr_acc, va_acc, fin_train_time - bgn_train_time, time.time() - eval_time))
            bgn_train_time = time.time()

        # Save model
        if iteration % 1000 == 0:
            save_out_dict = {'iteration': iteration, 
                             'state_dict': model.state_dict(), 
                             'optimizer': optimizer.state_dict()}
            save_out_path = os.path.join(models_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            print('Save model to {}'.format(save_out_path))

        # Move data to GPU
        load_time = time.time()
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        
        if False:
            print('Load data time: {}'.format(time.time() - load_time))
        
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
            raise Exception('Incorrect output_type!')
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if False:
            print(model.fc2.weight.data)
            print(model.fc2.weight.grad.data)
            pause

        if False:
            print('Train time: {}'.format(time.time() - forward_time))
        
        iteration += 1
        
            
def inference(args):
    
    # Arguments & parameters
    workspace = args.workspace
    output_type = args.output_type
    iteration = args.iteration
    cuda = args.cuda
    
    batch_size = 500
    
    models_dir = os.path.join(workspace, 'models')
    
    generator = TestDataGenerator(batch_size)

    # Load model
    model = DNN(output_type)
    checkpoint = torch.load(os.path.join(models_dir, 'md_{}_iters.tar'.format(iteration)))
    model.load_state_dict(checkpoint['state_dict'])
    
    if cuda:
        model.cuda()

    # Inference
    generate_func = generator.generate_test()
    dict = forward(model, generate_func, cuda, return_target=False)
    outputs = dict['output']
    
    predictions = np.argmax(outputs, axis=-1)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', default='')
    parser_train.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'])
    parser_train.add_argument('--output_type', default='softmax', choices=['softmax', 'sigmoid'])
    parser_train.add_argument('--lr', type=float, default=1e-3)
    parser_train.add_argument('--freeze', action='store_true', default=False)
    parser_train.add_argument('--resume_model_path', type=str, default='')
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', default='')
    parser_inference.add_argument('--output_type', default='softmax', choices=['softmax', 'sigmoid'])
    parser_inference.add_argument('--iteration', type=str, required=True)
    parser_inference.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    print(args)
    
    if args.mode == 'train':        
        train(args)
        
    elif args.mode == 'inference':
        inference(args)
        
    else:
        raise Exception('Error!')
        
        
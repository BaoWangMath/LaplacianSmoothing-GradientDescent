#------------------------------------------------------------------------------
# System module.
#------------------------------------------------------------------------------
import os
import random
import time
import copy
import argparse
import sys

#------------------------------------------------------------------------------
# Torch module.
# We used torch to build the WNLL activated DNN. Note torch utilized the
#dynamical computational graph, which is appropriate for our purpose, since
#in our model, we involves nearest neighbor searching, which is too slow by
#symbolic computing.
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

#------------------------------------------------------------------------------
# Numpy module.
#------------------------------------------------------------------------------
import numpy as np
import numpy.matlib

#------------------------------------------------------------------------------
# DNN and WNLL module
#------------------------------------------------------------------------------
from LeNet import *
from utils import *

#------------------------------------------------------------------------------
# LS-SGD Optimizer 
#------------------------------------------------------------------------------
from SGD import *
from LS_SGD import *


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    global best_acc
    best_acc = 0
    start_epoch = 0
    
    #--------------------------------------------------------------------------
    # Load the MNIST data
    #--------------------------------------------------------------------------
    print('==> Preparing data...')
    root = '../data'
    download = True
    
    trans=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
    
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans, download=download)
    
    # Convert the data into appropriate torch format.
    kwargs = {'num_workers':1, 'pin_memory':True}
    
    batchsize_test = len(test_set)/2
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs
                                             )
    
    batchsize_train = 4 # Set batchsize here!
    print('Batch size of the train set: ', batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batchsize_train,
                                               shuffle=True, **kwargs
                                              )
    
    #--------------------------------------------------------------------------
    # Build the model
    #--------------------------------------------------------------------------
    net = LeNet1().cuda()
    sigma = 1.5
    lr = 0.01
    weight_decay = 5e-4
    optimizer = Grad_SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    
    nepoch =200
    for epoch in xrange(nepoch):
        print('Epoch ID: ', epoch)
        #----------------------------------------------------------------------
        # Training
        #----------------------------------------------------------------------
        if epoch >=40 and (epoch//40 == epoch/40.0):
            lr = lr/5
            print("Descrease the Learning Rate, lr = ", lr)
            #optimizer = Grad_SJO_SGD(net.parameters(), lr=lr, sigma = sigma, momentum=0.9, nesterov=True)
            optimizer = Grad_SJO_SGD(net.parameters(), lr=lr, sigma = sigma, momentum=0.9, weight_decay=weight_decay, nesterov=True)
            #optimizer = Grad_SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        
        correct = 0; total = 0; train_loss = 0
        net.train()
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x.cuda()), Variable(target.cuda())
            score, loss = net(x, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.data[0]
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
                
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
        #----------------------------------------------------------------------
        # Testing
        #----------------------------------------------------------------------
        test_loss = 0; correct = 0; total = 0
        net.eval()
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
            score, loss = net(x, target)
            
            test_loss += loss.data[0]
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        #----------------------------------------------------------------------
        # Save the checkpoint
        #----------------------------------------------------------------------
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving model...')
            state = {
                'net': net,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint_MNIST'):
                os.mkdir('checkpoint_MNIST')
            torch.save(state, './checkpoint_MNIST/ckpt.t7')
            best_acc = acc
    
    print('The best acc: ', best_acc)

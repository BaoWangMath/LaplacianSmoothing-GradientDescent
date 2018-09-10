"""
LeNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])

class LeNet1(nn.Module):
    """
    Wider LeNet
    """
    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv1_drop = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        #self.conv2_drop = nn.Dropout(0.25)
        
        self.fc = nn.Linear(4*4*50, 512)
        #self.fc_drop = nn.Dropout(0.25)
        
        self.linear = nn.Linear(512, 10)
        
        self.loss = nn.CrossEntropyLoss()
    
    
    def forward(self, x, target):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        
        # Flatten the tensor
        x = x.view(-1, 4*4*50)
        
        x = self.fc(x)
        
        x = self.linear(x)
        target = target.long()
        loss = self.loss(x, target)
        return x, loss

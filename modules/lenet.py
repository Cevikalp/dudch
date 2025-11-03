#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:39:44 2021

@author: mlcv
"""
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_1_bn=nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv1_2_bn=nn.BatchNorm2d(32)
        
        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_1_bn=nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.LeakyReLU(0.1)
        self.conv2_2_bn=nn.BatchNorm2d(64)
        
        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_1_bn=nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.conv3_2_bn=nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128*4*4, 2)
        self.prelu_fc1 = nn.PReLU()
        self.feat_dim = 2
        self.fc2 = nn.Linear(2, num_classes)

    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.prelu1_1(self.conv1_1(x))
        x = self.conv1_1_bn(x)
        x = self.prelu1_2(self.conv1_2(x))
        x = self.conv1_2_bn(x)
        x = F.max_pool2d(x, 2)
        
        x = self.prelu2_1(self.conv2_1(x))
        x = self.conv2_1_bn(x)
        x = self.prelu2_2(self.conv2_2(x))
        x = self.conv2_2_bn(x)
        x = F.max_pool2d(x, 2)
        
        x = self.prelu3_1(self.conv3_1(x))
        x = self.conv3_1_bn(x)
        x = self.prelu3_2(self.conv3_2(x))
        x = self.conv3_2_bn(x)
        x = F.max_pool2d(x, 2)
        
        x = x.view(batch_size, -1)
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)

        return x, y

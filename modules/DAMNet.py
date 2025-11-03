#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 22:53:53 2022

@author: mlcv
"""
import math
import torch
import torchvision.models as models
import torch.nn as nn
from modules.lenet import LeNet
# from lenet import LeNet

   
class DAMSequential(nn.Module):
    def __init__(self, input_dim = 128, output_dim=10):
        super(DAMSequential, self).__init__()
        #self.model = models.resnet50(pretrained=True)
        self.model = LeNet(10)
        self.layers = []
        self.layers.append(nn.Linear(input_dim, int(output_dim*0.5)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(int(output_dim*0.5), output_dim))            
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(output_dim, output_dim))    
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        feats, out = self.model(x)
        out = self.layers(feats)
        return out
    
class DAMDebug(nn.Module):
    def __init__(self, input_dim = 2, output_dim=10):
        super(DAMDebug, self).__init__()
        #self.model = models.(pretrained=True)
        self.model = LeNet(10)
        self.layer1 = nn.Linear(input_dim, int(output_dim*0.5))
        self.layer1_act = nn.ReLU()
        self.layer2 = nn.Linear(int(output_dim*0.5), output_dim)  
        self.layer2_act = nn.ReLU()
        self.layer3 = nn.Linear(output_dim, output_dim) 

    def forward(self, x):
        feats, out = self.model(x)
        out = self.layer1(feats)
        out = self.layer1_act(out)
        out = self.layer2(out)
        out = self.layer2_act(out)
        out = self.layer3(out)
        return out
  
# x = torch.ones(128,2048)
# dam = DAM(x.shape[1], 256)
# out = dam(x)

class DAMGeneral(nn.Module):
     def __init__(self, input_dim = 2, output_dim=10, num_layers=2):
         super(DAMGeneral, self).__init__()
         # self.model = models.resnet50(pretrained=True)
         # input_dim = self.model.fc.weight.shape[1]
         self.model = LeNet(10)
         input_dim = self.model.fc2.weight.shape[1]
         self.out_dim = output_dim
         ratio = max(input_dim/output_dim, output_dim/input_dim)
         self.coef = ratio**(1/num_layers)
         self.layers = []
         
         for i in range(num_layers):
             if input_dim*self.coef > self.out_dim:
                 self.layers.append(nn.Linear(input_dim, output_dim))
                 input_dim = self.out_dim
             else:
                 self.layers.append(nn.Linear(input_dim, math.ceil(input_dim*self.coef)))
                 input_dim = math.ceil(input_dim*self.coef)
             self.layers.append(nn.PReLU())
             self.layers.append(nn.BatchNorm1d(input_dim,eps=1e-05))
         else:
             self.layers.append(nn.Linear(input_dim, output_dim))

         self.layers = nn.Sequential(*self.layers)
         
     def forward(self, x):
         feats, out = self.model(x)
         dam_out = self.layers(feats)
         return dam_out


class DAMEmb(nn.Module):
    def __init__(self, input_dim = 2, output_dim=10, num_layers=2):
        super(DAMEmb, self).__init__()
        #self.model = models.(pretrained=True)
        self.model = LeNet(10)
        input_dim = self.model.fc2.weight.shape[1]
        self.emb = nn.Embedding(int(input_dim), int(output_dim))
        self.layer1 = nn.Linear(output_dim, output_dim)
        self.layer1_act = nn.ReLU()
        self.layer2 = nn.Linear(int(output_dim), output_dim)  

    def forward(self, x):
        feats, out = self.model(x)
        feats = self.emb(feats)
        out = self.layer1(feats)
        out = self.layer1_act(out)
        out = self.layer2(out)
        return out
  

# x = torch.ones(128,3,112,112)
# dam = DAMGeneral(10, 4096, 3)
# print(dam)
# out = dam(x)
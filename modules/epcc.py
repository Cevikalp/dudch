import torch
import torch.nn as nn
import numpy as np

from modules.findcenters import FindCenters

class EPCC_Layer(nn.Module):
    def __init__(self,input_dim, device_name=None):
        super(EPCC_Layer,self).__init__()
        self.input_dim = input_dim
        self.device = device_name
        self.w = nn.Linear(self.input_dim,1,bias=False).to(self.device)
        self.wabs = nn.Linear(self.input_dim,1,bias=False).to(self.device)
        self.bias = nn.Parameter(torch.zeros(1),requires_grad=True).to(self.device)
        self.centers = nn.Parameter(torch.rand(1,self.input_dim)).to(self.device)
    def forward(self, inputs):# we do not need labels here
        #self.centers = self.centers.to(device)
        inputs_c =(inputs-self.centers)
        epcc_out = self.w(inputs_c) + self.wabs(inputs_c.abs())+self.bias.clamp(min=0.0)
        return epcc_out


class hinge(nn.Module):
    def __init__(self,margin=1.0):
        super(hinge,self).__init__()
        self.margin = margin
    def forward(self,input1,target):
        loss = (-target * input1.t() + self.margin).clamp(min=0)
        # loss = torch.pow(loss,2) 
        # loss += args.c * torch.mean(model.fc.weight ** 2)  # l2 penalty
        return loss.mean()
    



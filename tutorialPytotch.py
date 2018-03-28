#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 09:36:49 2018

@author: andrey
"""
#%% matrix
#init
import torch

x = torch.Tensor(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)
print(x.size())

#addition

y = torch.rand(5, 3)
print(x + y)

print(torch.add(x,y))

res= torch.Tensor(5,3)
torch.add(x,y, out =res)
print(res)


y.add_(x)
print(y)

#%% #resize
import torch
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
#%% numpy bridge
#convert torch to numpy
import torch
a = torch.ones(5)
print(a)
b= a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

#convert numpy to torch
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

#cuda 

flag = torch.cuda.is_available()
print(flag)
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y

#%% backprop
import torch
from torch.autograd import Variable



y = x  * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)


gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)
 

 
   


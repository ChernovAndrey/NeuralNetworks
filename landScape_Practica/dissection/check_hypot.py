#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:50:45 2018

@author: andrey
"""
#%%
from scipy.stats import mannwhitneyu,fligner
from landScape_Practica.myUtils import readData,saveData

import numpy as np
num_samples= 100
num_layers=14
res0 =readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res'+str(num_layers)+'_0')
res1 =readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res'+str(num_layers)+'_1')
#res0 =readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/result1','res13_0')
#res1 =readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/result1','res13_1')
print(res0.shape)
print(res1.shape)
p_val_shift=np.empty(shape=(num_samples))
p_val_varian=np.empty(shape=(num_samples))
for  i in range(num_samples):
    _,p_val_shift[i] =mannwhitneyu(res0[i].reshape(-1), res1[i].reshape(-1))
    _,p_val_varian[i] = fligner(res0[i].reshape(-1),res1[i].reshape(-1))
    print("p_value shift= ",p_val_shift[i])
    print("p_value varian= ",p_val_varian[i])
saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value','p_val_shift'+str(num_layers),p_val_shift)    
saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value','p_val_variance'+str(num_layers),p_val_varian)    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:25:15 2018

@author: andrey
"""

#%% generate random samples for check hypot
from myUtils import readData,saveData
import numpy as np
def gen_rand_samples_for_hypot(num_layers,len_dim_input=2,num_repeat=100, num_samples = 5000):# len_dim_input(написано только для 2 и 4)
                                                                                              #- количество размерностей у вхожного вектора
#    res0=np.empty(shape=0,dtype=float)
#    res1=np.empty(shape=0,dtype=float)
    allRes0 = readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/allResult',
                         'res'+str(num_layers)+'_'+'0') 
    allRes1 = readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/allResult',
                         'res'+str(num_layers)+'_'+'1') 
    
    half_num_samples = num_samples//2
    if (len_dim_input==2):
        res0 = np.empty((num_repeat,half_num_samples, allRes0.shape[1] ),dtype=float)  
        res1 = np.empty((num_repeat,half_num_samples, allRes1.shape[1] ),dtype=float)  
    if (len_dim_input==4):
        res0 = np.empty((num_repeat,half_num_samples, allRes0.shape[1],allRes0.shape[2],allRes0.shape[3] ),dtype=float)  
        res1 = np.empty((num_repeat,half_num_samples, allRes1.shape[1],allRes0.shape[2],allRes0.shape[3] ),dtype=float)  
        
    for i in range(num_repeat):
        rand_num0 = np.random.randint(allRes0.shape[0], size=half_num_samples)     # 60000- всего экземпляров
        rand_num1 = np.random.randint(allRes1.shape[0], size=half_num_samples)     # 60000- всего экземпляров
        for j in range(half_num_samples):
                k0 = rand_num0[j]
                k1 = rand_num1[j]
                res0[i][j] = allRes0[k0]
                res1[i][j] = allRes1[k1]
    print(res0.shape)
    print(res1.shape)
    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res'+str(num_layers)+'_0',res0)
    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res'+str(num_layers)+'_1',res1)
    


#%%
gen_rand_samples_for_hypot(12,2,100,5000) # len_dim :14-1; 13,12 - 2; остальные - 4;
    
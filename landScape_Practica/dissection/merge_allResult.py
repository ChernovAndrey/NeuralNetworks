#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:46:55 2018

@author: andrey
"""
from myUtils import readData,saveData
import numpy as np
num_layers=0
res= readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/result'+str(1),
                                          'res'+str(num_layers)+'_'+'1')

for i in range(2,7):
    loc_res= readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/result'+str(i),
                                          'res'+str(num_layers)+'_'+'1')
    print(loc_res.shape)
    res=np.concatenate( (res, loc_res) )
print(res.shape)    
saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/allResult','res'+str(num_layers)+'_1',res)

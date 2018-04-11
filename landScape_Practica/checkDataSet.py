#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:16:57 2018

@author: andrey
"""

#%%
import h5py
h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/2/ready_data.hdf5','r')
data = h5f['dataset_3000'][...]
h5f.close()


h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/2/ready_res.hdf5','r')
result = h5f['dataset_3000'][...]
h5f.close()

print(data.shape)
print(result.shape)

count_train=300000
x_train=data[:count_train]
x_test=data[count_train:]
y_train=result[:count_train]
y_test=result[count_train:]

#%%
import numpy as np
from neuralNetworks.landScape_Practica.IterativeAlgorithm import calculateResult
def getPoints(matrix): #32*32
    p=np.zeros(shape=(2,2))
    k=0 # индекс по массиву p
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] ==1:
                p[k]=(i,j)
                k+=1
#    print(p)
    return p
def checkOneEl(input): # input 32*32*2; answerExpec: 1
    input = input.reshape(2,32,32)
    matrix = input[0]
    p = getPoints(input[1])
    p1 = (p[0][0], p[0][1], matrix[p[0][0]][p[0][1]])
    p2 = (p[1][0], p[1][1], matrix[p[1][0]][p[1][1]])
    
    return calculateResult(p1,p2,input[0])
def checkDataset(data,ansExpec): # проверим каждый десятый элемент
    coef = 10
    n = len(data)/coef
    print(n)
    for i in range( int(n) ):
        if (i%1000 == 0):
            print("check number ",10*i)
        j = 10*i
        ans = checkOneEl(data[j])
        if ( int(ans) != int(ansExpec[j]) ):
            print("NOT MATCH")

#%%
checkDataset(data,result)


print(result)    




#%%
def mixDataset(data,result):
    for i in 10000:
        j = np.random.randint(data.shape[0])
        result[i], result[j]  = result[j], result[i] 
        data[i], data[j]  = data[j], data[i] 




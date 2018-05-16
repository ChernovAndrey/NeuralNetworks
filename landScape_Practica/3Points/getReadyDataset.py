#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 19:14:19 2018

@author: andrey
"""
#%%
count_landscapes=3000
count_tuples=15# количество комбанаций в данном случае трех точек на одном ландшафте
count_shifts=8
count_pixels=32
count_points = 3

import numpy as np

def readData(path,name):
    import h5py
    h5f = h5py.File(path,'r')
    result = h5f[name][...]
    h5f.close()
    return result

def saveData(path,name,data):
    import h5py
    h5f = h5py.File(path, 'w')
    h5f.create_dataset(name, data=data,dtype=np.float32)
    h5f.close()
#%%    
first_dim=count_landscapes*count_tuples*count_shifts
#%%
data  = readData('dataset.hdf5','input')
print(data.shape)
#%%
data = np.reshape(data,(first_dim,4,count_pixels,count_pixels))
print(data.shape)
data = np.reshape(data,(first_dim,count_pixels,count_pixels,4))
print(data.shape)

#%%
result  = readData('result.hdf5','result')
print(result.shape)
#%%
result = np.reshape(result,(first_dim,3))
print(result.shape)
#%%
reject  = readData('reject.hdf5','reject')
print(reject.shape)
#%%
print(result.shape)
print(reject.shape)
#%%
print(reject[0])
print(result[0])
#%%
#%%
reject = np.reshape(reject,(first_dim,3))
print(reject.shape)
#%%

print(data.shape)
print(result.shape)

count_train=300000
x_train=data[:count_train]
x_test=data[count_train:]
y_train=result[:count_train]
y_test=result[count_train:]
rej_train=reject[:count_train]
rej_test=reject[count_train:]
#%%
#ошибка опять с create_data
saveData('ready3PointTrainData.hdf5','trainData',x_train)
saveData('ready3PointTestData.hdf5','testData',x_test)

saveData('ready3PointTrainResult.hdf5','trainResult',y_train)
saveData('ready3PointTestResult.hdf5','testResult',y_test)

saveData('ready3PointTrainReject.hdf5','trainReject',rej_train)
saveData('ready3PointTestReject.hdf5','testReject',rej_test)





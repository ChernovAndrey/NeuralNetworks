#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 17:30:10 2018

@author: andrey
"""
def getTestData2Points(): 
    import numpy as np
    import h5py
    h5f = h5py.File('/home/andrey/datasetsNN/landScapes/landScape_3000_32/2/ready_data.hdf5','r')
    data = h5f['dataset_3000'][...]
    h5f.close()


    h5f = h5py.File('/home/andrey/datasetsNN/landScapes/landScape_3000_32/2/ready_res.hdf5','r')
    result = h5f['dataset_3000'][...]
    h5f.close()

 #   print(data.shape)
#    print(result.shape)

    count_train=300000
#x_train=data[:count_train]
    x_test=data[count_train:]
#y_train=result[:count_train]
    y_test=result[count_train:]
    return x_test,y_test


def saveData(path,name,data):
    import h5py
    h5f = h5py.File(path, 'a')
    h5f.create_dataset(name, data=data)
    h5f.close()
    
def readData(path,name_dataset):
    import h5py
    h5f = h5py.File(path,'r')
    data = h5f[name_dataset][...]
    h5f.close()
    return data    
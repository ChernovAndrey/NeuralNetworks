#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 17:30:10 2018

@author: andrey
"""
import numpy as np
import h5py as h5

work_dir_getTestData2Points = 'mix/trainData.hdf5'


def saveData(path,name,data):
    h5f = h5.File(path, 'a')
    h5f.create_dataset(name, data=data)
    h5f.close()
    
def readData(path,name_dataset):
    h5f = h5.File(path,'r')
    data = h5f[name_dataset][...]
    h5f.close()
    return data

def getTestData2Points(): #данные для двух точек
    x_train = readData(work_dir_getTestData2Points+'trainData.hdf5','train_3000_')
    print('shape x train =',x_train.shape)
    
    y_train = readData(work_dir_getTestData2Points+'trainResult.hdf5','result_3000')
    print('shape y train =',x_train.shape)
    
    x_test = readData(work_dir_getTestData2Points+'testData.hdf5','test_3000_')
    print('shape x test =',x_train.shape)
    
    y_test= readData(work_dir_getTestData2Points+'testResult.hdf5','resultTest_3000')
    print('shape y test =', x_train.shape)
    return x_train,y_train,x_test,y_test    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 17:30:10 2018

@author: andrey
"""
import numpy as np
import h5py as h5

work_dir_getTestData2Points = '/home/andrey/datasetsNN/landScapes/landScape_3000_32/mix/'


def saveData(path,name,data):
    h5f = h5.File(path, 'a')
    h5f.create_dataset(name, data=data)
    h5f.close()
    
def readData(path,name_dataset,read_all=True,begin_el=0,last_el=0):# begin_el -  с какого элемента считывать, last_el - до какого
    h5f = h5.File(path,'r')
    if read_all == True:
        data = h5f[name_dataset][...]
    else:
        data = h5f[name_dataset][begin_el:last_el]
    h5f.close()
    return data
#
#def getTestData2Points(): #данные для двух точек
#    x_train = readData(work_dir_getTestData2Points+'trainData.hdf5','train_3000_')
#    print('shape x train =',x_train.shape)
#    
#    y_train = readData(work_dir_getTestData2Points+'trainResult.hdf5','result_3000')
#    print('shape y train =',y_train.shape)
#    
#    x_test = readData(work_dir_getTestData2Points+'testData.hdf5','test_3000_')
#    print('shape x test =',x_test.shape)
#    
#    y_test= readData(work_dir_getTestData2Points+'testResult.hdf5','resultTest_3000')
#    print('shape y test =', y_test.shape)
#    return x_train,y_train,x_test,y_test    
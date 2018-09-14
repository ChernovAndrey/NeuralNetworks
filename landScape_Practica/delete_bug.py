#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:25:21 2018

@author: andrey
"""
#%%
import numpy as np
from myUtils import readData, saveData

path_to_dataset = 'dataset_11_09_2018'
save_path = 'dataset_clear_11_09_2018.hdf5'
#path_to_dataset = save_path
name_y_test = 'y_test'
name_x_test = 'X_test'
name_rej_test = 'rej_test'

name_y_train = 'y_train'
name_x_train = 'X_train'
name_rej_train = 'rej_train'


def check_bug(rej, y):  # проверка датасета на ошибки
    bugEl1 = np.where(((rej < 0) & (y == 1)))[0]  
    print(bugEl1)
    print('len bug1 = ', len(bugEl1))
#    print(rej_test[bugEl1[0]], y_test[bugEl1[0]])
#
    bugEl2 = np.where(((rej > 0) & (y == 0)))[0]  
    print('len bug 2 = ', len(bugEl2))
    return np.concatenate((bugEl1, bugEl2))
#    print(rej_test[bugEl2[0]], y_test[bugEl2[0]])

def delete_bug(X, y, rej, p): # p - номера неверных данных
    X = np.delete(X,p, axis=0)
    y = np.delete(y,p)
    rej = np.delete(rej,p)
    print('X shape ', X.shape)
    print('y shape ', y.shape)
    print('rej shape ', rej.shape)
    return X, y, rej
def save_clear_data(X_train, y_train, X_test, y_test, rej_train, rej_test):
    saveData(save_path, name_x_train, X_train)
    saveData(save_path, name_y_train, y_train)
    saveData(save_path, name_rej_train, rej_train)
    
    saveData(save_path, name_x_test, X_test)
    saveData(save_path, name_y_test, y_test)
    saveData(save_path, name_rej_test, rej_test)
        
#%%
    
X_test = readData(path_to_dataset, name_x_test)
y_test = readData(path_to_dataset, name_y_test)
rej_test = readData(path_to_dataset, name_rej_test)
print('x_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)
print('rej_test shape: ', rej_test.shape)    

X_train = readData(path_to_dataset, name_x_train)
y_train = readData(path_to_dataset, name_y_train)
rej_train = readData(path_to_dataset, name_rej_train)
print('x_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('rej_train shape: ', rej_train.shape)

p = check_bug(rej_test, y_test)
print('all len bug test = ', len(p))
X_test, y_test, rej_test = delete_bug(X_test, y_test, rej_test, p)

p = check_bug(rej_train, y_train)
print('all len bug train = ', len(p))
X_train, y_train, rej_train = delete_bug(X_train, y_train, rej_train, p)    

#save_clear_data(X_train, y_train, X_test, y_test, rej_train, rej_test)




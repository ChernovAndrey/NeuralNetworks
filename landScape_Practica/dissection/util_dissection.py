#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 19:05:14 2018

@author: andrey
"""

#%%
from keras.models import load_model
from myUtils import readData,saveData
from keras_utils import get_layer_output
import numpy as np
from config import *
import os


def getInd2Class(y_test): #разбиваем x_test на два класса, первый где y_test=0, второй, где y_test=1, возвращаем индексы
    ind0 = np.where(y_test==0)[0]
    ind1 = np.where(y_test==1)[0]
    return ind0,ind1
 

#сохраняет разделенные(по какому-либо признаку) результаты для нужных слоев 
#path_x_test - список из двух строк. Первая строка название файла, вторая название датасета(path_y_test - аналогично).
#n_layers - номера слоев, которые нужно исследовать
#f_divided  - функция, которая делит данные( по меткам). Возращает индексы двух групп.
def save_divided_output_layers(work_dir,name_model,path_x_test,path_y_test,f_divided, begin_el, end_el,nFiles, n_layers='all'): # читаем не все, так как может быть out of memmory
    model = load_model(work_dir+name_model)                                                                                     
    x_test =readData(*path_x_test,False,begin_el,end_el)
    y_test =readData(*path_y_test,False,begin_el,end_el)
    print('finish read data')
    output = get_layer_output(model,x_test,n_layers)
    print('finish get output layer')
    ind0, ind1 = f_divided(y_test)
    if n_layers == 'all':
        n_layers = range(0,len(model.layers))
    for i in n_layers:
        res0 = output[i][ind0] #где y_tes=0
        res1 = output[i][ind1]#где y_test=1
        saveData(os.path.join(work_dir , file_save_divided) ,'out_n' + str(nFiles) +'_' + str(i)+'_0',res0)        
        saveData(os.path.join(work_dir , file_save_divided),'out_n' + str(nFiles) +'_' +  str(i)+'_1',res1)
    print('finish function save_divided_output_layers')      
    return n_layers 

#count_class - количество классов(реализовано для двух и одного). Если их два то будет в конце файлов приписка '_0' и '_1'. Если один то ничего не будет.
def merge_result(count_layers,count_files,name_file, load_name_dataset,save_name_dataset, count_class=2): # load_name_dataset -  общая часть от имени датасетов(напрмиер 'out_n')
    for i in range(count_layers):                                                           # save_name_dataset - куда созранять
        
        if count_class == 2:
            res0 = readData(os.path.join(work_dir , name_file) , load_name_dataset + str(0) +'_' + str(i)+'_0')        
            res1 = readData(os.path.join(work_dir , name_file), load_name_dataset + str(0) +'_' + str(i)+'_1')
            print(res0.shape)
            print(res1.shape)
            for j in range(1, count_files):
                res0 = np.concatenate((res0,readData(os.path.join(work_dir , name_file) ,load_name_dataset + str(j) +'_' + str(i)+'_0')) , axis=0)        
                res1 = np.concatenate((res1, readData(os.path.join(work_dir , name_file),load_name_dataset + str(j) +'_' + str(i)+'_1')) , axis=0)
            print(res0.shape)
            print(res1.shape)
            saveData(os.path.join(work_dir , name_file) , save_name_dataset + str(i)+'_0',res0)        
            saveData(os.path.join(work_dir , name_file) , save_name_dataset + str(i)+'_1',res1)
        
        if count_class == 1:
            res = readData(os.path.join(work_dir , name_file) , load_name_dataset + str(0) +'_' + str(i))        
            print(res.shape)
            for j in range(1, count_files):
                res = np.concatenate((res,readData(os.path.join(work_dir , name_file) ,load_name_dataset + str(j) +'_' + str(i))) , axis=0)        
            print(res.shape)
            saveData(os.path.join(work_dir , name_file) , save_name_dataset + str(i),res)

        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:41:34 2018

@author: andrey
"""
#%%
import os
import numpy as np
def getInd2Class(y_test): #разбиваем x_test на два класса, первый где y_test=0, второй, где y_test=1, возвращаем индексы
    ind0 = np.where(y_test==0)[0]
    ind1 = np.where(y_test==1)[0]
    return ind0,ind1

work_dir = '' # заполнить!
name_model = os.path.join(work_dir,'../model_AlexNet_fullConv550_01_08_2018.hdf5')
path_x_test = [os.path.join(work_dir,'../dataset_30_07_2018'), 'x_test'] # первый элемент  - это файл; второй имя датасета
path_y_test = [os.path.join(work_dir,'../dataset_30_07_2018'), 'y_test'] # первый элемент  - это файл; второй имя датасета
f_divided = getInd2Class # функция разделения на классы
 
file_save_divided = 'out_layers' # имя файла, куда сохранять функции save_divided_output_layers
file_save_gen_rand = 'rand_samples_result' # имя файла, куда сохранять функции gen_rand_samples_for_hypot
file_save_pvalue_global = 'p-value_global'

count_thr = 5
num_repeat=100
num_samples = 5000
count_test_samples = 60000
n_layers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] # номера словев, которые надо исследовать
#n_layers = 'all'


flag_by_neuron = True # выолнить ли исследование по нейронно

file_save_by_neuron = 'p-value_by_neuron'


flag_stat_moments = False # посчитать статические моменты

file_save_mean0 = 'mean0' # для двух классов
file_save_mean1 = 'mean1' 
file_save_var0 = 'var0' 
file_save_var1 = 'var1' 


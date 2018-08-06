#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:08:03 2018

@author: andrey
"""
# файл в котором все выполняется. ( то есть генерация образцов, их проверка и т д.)
#%%
from util_dissection import save_divided_output_layers, merge_result
from multiprocessing import Pool
import os
import numpy as np
from myUtils import saveData,readData
from config import *
from check_hypot import check_hypot, check_hypot_batches, get_statistical_moments, check_hypot_by_neuron_for_matrix, check_hypot_by_neuron_for_vectors

def gen_rand_samples_for_hypot(num_layers,nFiles, num_repeat=100, num_samples = 5000):# len_dim_input(написано только для 2 и 4)
                                                                                              #- количество размерностей у вхожного вектора
#    res0=np.empty(shape=0,dtype=float)
#    res1=np.empty(shape=0,dtype=float)
    in_res0 = readData(os.path.join(work_dir,file_save_divided), 
                         'out'+str(num_layers)+'_'+'0') 
    in_res1 = readData( os.path.join(work_dir,file_save_divided),
                         'out'+str(num_layers)+'_'+'1') 
    
    
    half_num_samples = num_samples//2
    out_res0 = np.empty( (num_repeat,half_num_samples, *in_res0.shape[1:]) ,dtype=float)  
    out_res1 = np.empty( (num_repeat,half_num_samples, *in_res1.shape[1:]) ,dtype=float) 
#    if (len_dim_input==2):
#        res0 = np.empty((num_repeat,half_num_samples, in_res0.shape[1] ),dtype=float)  
#        res1 = np.empty((num_repeat,half_num_samples, in_res1.shape[1] ),dtype=float)  
#    if (len_dim_input==4):
#        res0 = np.empty((num_repeat,half_num_samples, in_res0.shape[1],in_res0.shape[2],in_res0.shape[3] ),dtype=float)  
#        res1 = np.empty((num_repeat,half_num_samples, in_res1.shape[1],in_res0.shape[2],in_res0.shape[3] ),dtype=float)  
#        
    for i in range(num_repeat):
        rand_num0 = np.random.randint(in_res0.shape[0], size=half_num_samples)     # 60000- всего экземпляров
        rand_num1 = np.random.randint(in_res1.shape[0], size=half_num_samples)     # 60000- всего экземпляров
        for j in range(half_num_samples):
                k0 = rand_num0[j]
                k1 = rand_num1[j]
                out_res0[i][j] = in_res0[k0]
                out_res1[i][j] = in_res1[k1]
    print(out_res0.shape)
    print(out_res1.shape)
    saveData(os.path.join(work_dir,file_save_gen_rand) , 'out_n' + str(nFiles) +'_' +  str(num_layers)+'_0',out_res0)
    saveData(os.path.join(work_dir,file_save_gen_rand) ,'out_n' + str(nFiles) +'_' + str(num_layers) + '_1',out_res1)
    

	  
 #%%
if __name__ == '__main__':
#    count_batches_tests = 5000
#    print(count_test_samples//count_batches_tests)
#    for i in range(count_test_samples//count_batches_tests): # не можем читать все так как много памяти то все занимает.
#        save_divided_output_layers(work_dir,name_model,path_x_test,path_y_test,f_divided,count_batches_tests*i,count_batches_tests*(i+1),i,n_layers)
#        print(i)
#    merge_result(len(n_layers),count_test_samples//count_batches_tests,file_save_divided,'out_n','out')
#    
    count_batches_repeats = 20
#    for i in n_layers:
#        for j in range(count_batches_repeats):
#            gen_rand_samples_for_hypot(i,j,num_repeat//count_batches_repeats,num_samples)
#    merge_result(len(n_layers),num_repeat//10,file_save_gen_rand,'out_n','out')
#    for i in n_layers:
#        for j in range(count_batches_repeats):
#            check_hypot_batches(i,5, j)   # так как в озу целиком все не влазит     
#    merge_result(len(n_layers),count_batches_repeats,'p-value_global','p_val_shiftn','p_val_shift',1)
#    merge_result(len(n_layers),count_batches_repeats,'p-value_global','p_val_variancen','p_val_variance',1)
#    
#    if flag_stat_moments == True:
#        for i in n_layers:
#            for j in range(count_batches_repeats):
#                print(i,j)
#                get_statistical_moments(i,5, j)   # так как в озу целиком все не влазит     

    if flag_by_neuron == True: # тут аккуратно, для разных слоев, вызваются разные функции.(лучше выполнять послойно)
        num_layers = 1
        for i in range(count_batches_repeats):
                 check_hypot_by_neuron_for_vectors(num_layers,i)
#                check_hypot_batches(i,5, j)   # так как в озу целиком все не влазит     
        
#%%
#merge_result(17,20,'var1','var_n','var',1)                 
        

    
#%%
 

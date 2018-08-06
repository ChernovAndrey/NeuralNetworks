#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:50:45 2018

@author: andrey
"""
#%%
num_layers = 0
import os
from scipy.stats import mannwhitneyu,fligner
from myUtils import readData,saveData
import numpy as np
from multiprocessing import Pool        
from config import * 


def check_hypot_batches(num_layers, num_repeat, nFiles):
    res0 = readData( os.path.join(work_dir,file_save_gen_rand) , 'out_n' + str(nFiles) +'_' +  str(num_layers)+'_0')
    res1 = readData(os.path.join(work_dir,file_save_gen_rand) ,'out_n' + str(nFiles) + '_' + str(num_layers) + '_1')
    
    print(res0.shape)
    print(res1.shape)
    p_val_shift=np.empty(shape=(num_repeat))
    p_val_varian=np.empty(shape=(num_repeat))
    for  i in range(num_repeat):
        _,p_val_shift[i] =mannwhitneyu(res0[i].reshape(-1), res1[i].reshape(-1))
        _,p_val_varian[i] = fligner(res0[i].reshape(-1), res1[i].reshape(-1))
        print("p_value shift= ",p_val_shift[i])
        print("p_value varian= ",p_val_varian[i])
#    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value','p_val_shift'+str(num_layers),p_val_shift)    
#    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value','p_val_variance'+str(num_layers),p_val_varian)    
    saveData(os.path.join(work_dir,file_save_pvalue_global),'p_val_shiftn' + str(nFiles) + '_' + str(num_layers),p_val_shift)    
    saveData(os.path.join(work_dir,file_save_pvalue_global),'p_val_variancen' + str(nFiles) + '_' + str(num_layers),p_val_varian)   


def get_statistical_moments(num_layers, num_repeat, nFiles):
    res0 = readData( os.path.join(work_dir,file_save_gen_rand) , 'out_n' + str(nFiles) +'_' +  str(num_layers)+'_0')
    res1 = readData(os.path.join(work_dir,file_save_gen_rand) ,'out_n' + str(nFiles) + '_' + str(num_layers) + '_1')
    
    print(res0.shape)
    print(res1.shape)
    res0_mean = np.empty(shape=(num_repeat))
    res1_mean = np.empty(shape=(num_repeat))
    res0_var =np.empty(shFape=(num_repeat))
    res1_var =np.empty(shape=(num_repeat))
    for  i in range(num_repeat):
        res0_mean[i] =np.mean(res0[i].reshape(-1))
        res1_mean[i] =np.mean(res1[i].reshape(-1))
        res0_var[i] = np.var(res0[i].reshape(-1), ddof=1)
        res1_var[i] = np.var(res1[i].reshape(-1), ddof=1)

    saveData(os.path.join(work_dir,file_save_mean0),'mean_n' + str(nFiles) + '_' + str(num_layers),res0_mean)    
    saveData(os.path.join(work_dir,file_save_mean1),'mean_n' + str(nFiles) + '_' + str(num_layers),res1_mean)   
    saveData(os.path.join(work_dir,file_save_var0),'var_n' + str(nFiles) + '_' + str(num_layers),res0_var)   
    saveData(os.path.join(work_dir,file_save_var1),'var_n' + str(nFiles) + '_' + str(num_layers),res1_var)   

    
def check_hypot(num_layers,num_repeat):
    res0 =readData(os.path.join(work_dir,file_save_gen_rand),'out'+str(num_layers)+'_0')
    res1 =readData(os.path.join(work_dir,file_save_gen_rand),'out'+str(num_layers)+'_1')

    print(res0.shape)
    print(res1.shape)
    p_val_shift=np.empty(shape=(num_repeat))
    p_val_varian=np.empty(shape=(num_repeat))
    for  i in range(num_repeat):
        _,p_val_shift[i] =mannwhitneyu(res0[i].reshape(-1), res1[i].reshape(-1))
        _,p_val_varian[i] = fligner(res0[i].reshape(-1), res1[i].reshape(-1))
        print("p_value shift= ",p_val_shift[i])
        print("p_value varian= ",p_val_varian[i])
#    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value','p_val_shift'+str(num_layers),p_val_shift)    
#    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value','p_val_variance'+str(num_layers),p_val_varian)    
    saveData(os.path.join(work_dir,file_save_pvalue_global),'p_val_shift' + str(num_layers),p_val_shift)    
    saveData(os.path.join(work_dir,file_save_pvalue_global),'p_val_variance' + str(num_layers),p_val_varian)   

def check_hypot_by_neuron_for_vectors(num_layers, nFiles=-1): # например для dense слоев; # для conv1d - тожк подходит
    
    res0 = readData( os.path.join(work_dir,file_save_gen_rand) , 'out_n' + str(nFiles) +'_' +  str(num_layers)+'_0')
    res1 = readData(os.path.join(work_dir,file_save_gen_rand) ,'out_n' + str(nFiles) + '_' + str(num_layers) + '_1')
 
    
    print("res0 shape =",res0.shape) 
#    p_val_shift=np.empty(shape=(num_samples))
    p_val_shift=np.empty(shape=(res0.shape[2],res0.shape[0]))# так как res.shape = 100*2500*128; 
#    p_val_varian=np.empty(shape=(num_samples))
    p_val_varian=np.empty(shape=(res0.shape[2],res0.shape[0]))
#    for  i in range(num_samples):
    print("p_val shape=",p_val_shift.shape)
    for i in range(p_val_shift.shape[0]):
        for j in range(p_val_shift.shape[1]):
           # print("shapes loc res=",res0[j,:,i])
            _,p_val_shift[i][j] = mannwhitneyu(res0[j,:,i].reshape(-1), res1[j,:,i].reshape(-1))
            _,p_val_varian[i][j] = fligner(res0[j,:,i].reshape(-1), res1[j,:,i].reshape(-1))
#    print("p_value shift= ",p_val_shift[i])
#    print("p_value varian= ",p_val_varian[i])
#    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value','p_val_shift'+str(num_layers),p_val_shift)    
#    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value','p_val_variance'+str(num_layers),p_val_varian)    
    saveData(file_save_by_neuron,'p_val_shiftn'+str(nFiles)+"_"+str(num_layers),p_val_shift)    
    saveData(file_save_by_neuron,'p_val_variancen'+str(nFiles)+"_"+str(num_layers),p_val_varian)   

def check_hypot_by_neuron_for_matrix(num_layers, nFiles=-1): # например для conv2d
#    num_samples = 1
#    res0 =readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res'+str(num_layers)+'_0')
#    res1 =readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res'+str(num_layers)+'_1')

    res0 = readData( os.path.join(work_dir,file_save_gen_rand) , 'out_n' + str(nFiles) +'_' +  str(num_layers)+'_0')
    res1 = readData(os.path.join(work_dir,file_save_gen_rand) ,'out_n' + str(nFiles) + '_' + str(num_layers) + '_1')
 
    
#    print(all_results[nFiles][0].shape)
#    print(all_results[nFiles][1].shape)
    print("res0 shape =",res0.shape) 
#    p_val_shift=np.empty(shape=(num_samples))
    p_val_shift=np.empty(shape=(res0.shape[2],res0.shape[3],res0.shape[0]))# так как res.shape = 100*2500*16*16*32 
#    p_val_varian=np.empty(shape=(num_samples))
    p_val_varian=np.empty(shape=(res0.shape[2],res0.shape[3],res0.shape[0]))
#    for  i in range(num_samples):
    print("p_val shape=",p_val_shift.shape)
    for i in range(p_val_shift.shape[0]):
        for j in range(p_val_shift.shape[1]):                        
            for k in range(p_val_shift.shape[2]):
                if(k%50==0):
                    print("start i,j,k =",i,j,k)
                if ( sum(res0[k,:,i,j,:].reshape(-1))==0 and (sum(res1[k,:,i,j,:].reshape(-1))==0)):
                    p_val_shift[i][j][k] = -1.0
                    p_val_varian[i][j][k] = -1.0
                _,p_val_shift[i][j][k] = mannwhitneyu(res0[k,:,i,j,:].reshape(-1), res1[k,:,i,j,:].reshape(-1))
                _,p_val_varian[i][j][k] = fligner(res0[k,:,i,j,:].reshape(-1), res1[k,:,i,j,:].reshape(-1))
#    print("p_value shift= ",p_val_shift[i])
#    print("p_value varian= ",p_val_varian[i])
#    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value','p_val_shift'+str(num_layers),p_val_shift)    
#    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value','p_val_variance'+str(num_layers),p_val_varian)    
    saveData(file_save_by_neuron,'p_val_shiftn'+str(nFiles)+"_"+str(num_layers),p_val_shift)    
    saveData(file_save_by_neuron,'p_val_variancen'+str(nFiles)+"_"+str(num_layers),p_val_varian)   



#%%    

#num_layers = 0    
#for i in range(count_thread):
#    res0 =  readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res_n'+str(i)+"_" + str(num_layers)+'_0')
#    res1 =  readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res_n'+str(i)+ str(num_layers)+'_1')
#    all_results[i] = [res0,res1]
#
#
#if __name__ == '__main__':
#    with Pool(count_thread) as p:
#        p.map(check_hypot,  (all_results)) 
#


#%%
#count_thread= 1
#all_results = [[] for i in range(count_thread)]


#for i in range(count_thread):
#    res0 =  readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res_n'+str(i)+"_" + str(num_layers)+'_0')
#    res1 =  readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res_n'+str(i)+ str(num_layers)+'_1')
#    all_results[0] = [res0,res1]
#    check_hypot_by_neuron_for_matrix(i)

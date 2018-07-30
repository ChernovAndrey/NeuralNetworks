#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:50:45 2018

@author: andrey
"""
#%%
num_layers = 0

from scipy.stats import mannwhitneyu,fligner
from myUtils import readData,saveData
import numpy as np
from multiprocessing import Pool        
count_thread = 1

all_results = [[] for i in range(count_thread)]

def check_hypot(nThread):
    num_layers =0
    num_samples = 10
#    res0 =readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res'+str(num_layers)+'_0')
#    res1 =readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res'+str(num_layers)+'_1')

#    res0 = readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res_n'+str(nThread)+"_" + str(num_layers)+'_0')
#    res1 = readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res_n'+str(nThread) + str(num_layers)+'_1')

    print(all_results[nThread][0].shape)
    print(all_results[nThread][1].shape)
    p_val_shift=np.empty(shape=(num_samples))
    p_val_varian=np.empty(shape=(num_samples))
    for  i in range(num_samples):
        res0 = all_results[nThread][0][i]
        res1 = all_results[nThread][1][i]
        _,p_val_shift[i] =mannwhitneyu(res0, res1)
        _,p_val_varian[i] = fligner(res0,res1)
        print("p_value shift= ",p_val_shift[i])
        print("p_value varian= ",p_val_varian[i])
#    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value','p_val_shift'+str(num_layers),p_val_shift)    
#    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value','p_val_variance'+str(num_layers),p_val_varian)    
    saveData('by_neuron/p-value','p_val_shift_n'+str(nThread)+"_"+str(num_layers),p_val_shift)    
    saveData('by_neuron/p-value','p_val_variance_n'+str(nThread)+"_"+str(num_layers),p_val_varian)   

def check_hypot_by_neuron_for_vectors(nThread=-1): # для dense слоев
#    num_samples = 1
#    res0 =readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res'+str(num_layers)+'_0')
#    res1 =readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res'+str(num_layers)+'_1')

#    res0 = readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res_n'+str(nThread)+"_" + str(num_layers)+'_0')
#    res1 = readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res_n'+str(nThread) + str(num_layers)+'_1')

#    print(all_results[nThread][0].shape)
#    print(all_results[nThread][1].shape)
    res0,res1 = all_results[nThread]
    print("res0 shape =",res0.shape) 
#    p_val_shift=np.empty(shape=(num_samples))
    p_val_shift=np.empty(shape=(res0.shape[2],res0.shape[0]))# так как res.shape = 100*2500*128; 
#    p_val_varian=np.empty(shape=(num_samples))
    p_val_varian=np.empty(shape=(res0.shape[2],res0.shape[0]))
#    for  i in range(num_samples):
    print("p_val shape=",p_val_shift.shape)
    for i in range(p_val_shift.shape[0]):
        for j in range(p_val_shift.shape[1]):
            print("shapes loc res=",res0[j,:,i])
            _,p_val_shift[i][j] = mannwhitneyu(res0[j,:,i], res1[j,:,i])
            _,p_val_varian[i][j] = fligner(res0[j,:,i], res1[j,:,i])
#    print("p_value shift= ",p_val_shift[i])
#    print("p_value varian= ",p_val_varian[i])
#    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value','p_val_shift'+str(num_layers),p_val_shift)    
#    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p-value','p_val_variance'+str(num_layers),p_val_varian)    
    saveData('by_neuron/p-value','p_val_shift_n'+str(nThread)+"_"+str(num_layers),p_val_shift)    
    saveData('by_neuron/p-value','p_val_variance_n'+str(nThread)+"_"+str(num_layers),p_val_varian)   

def check_hypot_by_neuron_for_matrix(nThread=-1): # то есть для всех слоев, кроме dense слоев
#    num_samples = 1
#    res0 =readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res'+str(num_layers)+'_0')
#    res1 =readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res'+str(num_layers)+'_1')

#    res0 = readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res_n'+str(nThread)+"_" + str(num_layers)+'_0')
#    res1 = readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res_n'+str(nThread) + str(num_layers)+'_1')

#    print(all_results[nThread][0].shape)
#    print(all_results[nThread][1].shape)
    res0,res1 = all_results[0]
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
    saveData('by_neuron/p-value','p_val_shift_n'+str(nThread)+"_"+str(num_layers),p_val_shift)    
    saveData('by_neuron/p-value','p_val_variance_n'+str(nThread)+"_"+str(num_layers),p_val_varian)   



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
count_thread= 1
all_results = [[] for i in range(count_thread)]


for i in range(count_thread):
    res0 =  readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res_n'+str(i)+"_" + str(num_layers)+'_0')
    res1 =  readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res_n'+str(i)+ str(num_layers)+'_1')
    all_results[0] = [res0,res1]
    check_hypot_by_neuron_for_matrix(i)
    





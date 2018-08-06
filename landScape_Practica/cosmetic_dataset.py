#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 19:16:52 2018

@author: andrey
"""
#%% когда y_test = 0; reject <0, если y_test = 1, reject > 0
from myUtils import readData,saveData
import numpy as np
import os
work_file = '/home/andrey/datasetsNN/landScapes/landScape_3000_32/mix_fixed/dataset_30_07_2018' 
save_work_file = '/home/andrey/datasetsNN/landScapes/landScape_3000_32/mix_fixed/dataset_07_08_2018' 
y_test=readData(work_file,'y_test')
x_test = readData(work_file,'x_test')
count_points = 2
count_thread = 4
nThread = np.arange(count_thread)
count_el_one_thread = len(x_test)//count_thread
if(len(x_test)%count_thread != 0):
    print('WARNING, len(x_test)%count_thread != 0 ')

print(x_test.shape)
print(y_test.shape)
#%% 
def separate_landscape_and_points(matrix):
    print(matrix.shape)
    mod_matrix = np.moveaxis(matrix,3,0)
    print(mod_matrix.shape)
    landscape = mod_matrix[0]
    coord_points = np.where(mod_matrix[1]==1)
    return landscape, coord_points[1], coord_points[2] # отдаем координаты x и y,причем парно(то есть первые две координаты принадлежат 0-ому ландшафтну)
#%% main get reject
from IterativeAlgorithm import calculateResult    
from multiprocessing import Pool
def get_reject_and_result(nThread): # для готовых данных
    begin_el = count_el_one_thread*nThread
    end_el = count_el_one_thread*(nThread+1)
    landscape, x_points, y_points = separate_landscape_and_points(x_test[begin_el:end_el])
    print(landscape.shape)
    print(x_points.shape)
    print(y_points.shape)
    reject = np.empty( count_el_one_thread )
    y_test = np.empty( count_el_one_thread )
    for i in range(begin_el,end_el):
        if i % 1000 == 0:
            print(i)
        y_test,reject[i] = calculateResult( np.array([ x_points[2*i], y_points[2*i], landscape[i,x_points[2*i], y_points[2*i]] ]),
              np.array([ x_points[2*i+1], y_points[2*i+1], landscape[i,x_points[2*i+1], y_points[2*i+1]] ]),landscape[i] )
    
    saveData(save_work_file,'y_test_n'+str(nThread),y_test)
    saveData(save_work_file,'reject_n'+str(nThread),reject)

def merge_result():
    y_test = readData(save_work_file,'y_test_n0')
    reject = readData(save_work_file,'reject_n0')
    for i in range(1,count_thread):
        y_test = np.concatenate((y_test,readData(save_work_file, 'y_test_n'+str(i))) , axis=0)
        reject = np.concatenate((reject,readData(save_work_file, 'reject_n'+str(i))) , axis=0)
    
    saveData(save_work_file,'y_test',y_test)
    saveData(save_work_file,'reject',reject)
if __name__ == '__main__':
#    saveData(save_work_file,'x_test',x_test)
    with Pool(count_thread) as p:
        p.map(get_reject_and_result, range(count_thread) )
    print('finish  get_reject_and_result')
    merge_result()
#%% evaluate reject
reject = readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/mix_fixed/dataset_30_07_2018','reject')
#%%
print(y_test)    
print(reject)    

#%%
ind_error0 = np.where( (y_test==1) & (reject<0))
ind_error1 = np.where( (y_test==0) & (reject>0))
#%%

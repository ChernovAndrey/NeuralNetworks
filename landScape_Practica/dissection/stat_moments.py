#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 19:38:08 2018

@author: andrey
"""

#%%
# сравнение моментов двух классов.
from myUtils import readData
name_moment = 'var'
#%%
num_layers = 0
work_file0 ='/home/andrey/datasetsNN/landScapes/landScape_3000_32/p_val_02_08_2018/'+name_moment +str(0)
work_file1 ='/home/andrey/datasetsNN/landScapes/landScape_3000_32/p_val_02_08_2018/' + name_moment +str(1)
import numpy as np
i  = 1 # num_layers
#for i in range(count_layers):
y0 = readData(work_file0,name_moment + str(i))
y1 = readData(work_file1,name_moment + str(i))
print('median=')
print(np.median(y0))
print(np.median(y1))
    
print('mean=')
print(np.mean(y0))
print(np.mean(y1))

print('var=') # 14,15 - слой
print(np.var(y0,ddof=1))
print(np.var(y1,ddof=1))


#%%
import matplotlib.pyplot as plt

count_layers = 17
x_list = [c  for c in range(count_layers)]
y_list_0 = [None] * count_layers
y_list_1 = [None] * count_layers


# stat - функция, которая применяется к данным для их сравнения, (например медиана или арифмет. среднее)
# flag_show_diff - рисовать ли график модуля разницы между первым и вторым классом
def show(stat,flag_show_diff=False): 
    for i in range(count_layers):
        y0 = readData(work_file0,name_moment + str(i))
        y1 = readData(work_file1,name_moment + str(i))        
        y_list_0[i] = stat(y0)
        y_list_1[i] = stat(y1)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('layer number')
    ax1.set_ylabel('values of statistics')
    ax1.plot(x_list, y_list_0, y_list_1)
    plt.show()
    if flag_show_diff == True:
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        ax2.set_xlabel('layer number')
        ax2.set_ylabel('difference between values of statistics')
        plt.plot(x_list, abs(np.asarray(y_list_0) - np.asarray(y_list_1)) )
        plt.show()
    print(abs(np.asarray(y_list_0) - np.asarray(y_list_1)))
show(np.median, True)        
    
#%%
x_list = [c  for c in range(17)]
print(x_list)    
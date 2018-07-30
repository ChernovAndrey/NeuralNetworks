#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 17:35:51 2018

@author: andrey
"""
#%%
from myUtils import readData
import numpy as np
import pylab as plt
import keras
num_layers  = 3
shift = readData("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p_value_by_neuron/p-value","p_val_shift_n-1_"+str(num_layers))
var  = readData("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p_value_by_neuron/p-value","p_val_variance_n-1_"+str(num_layers))
if num_layers>=12:
    shift = np.median(shift,axis=1)
    var = np.median(var,axis=1)
else:
    shift = np.median(shift,axis=2)
    var = np.median(var,axis=2)
print(shift.shape)
print(var.shape)
from matplotlib import colors
def show_pictures(arrays,name='',save_pictures=False): #name - название момента по которому проводится оценка.
    if num_layers == 13:
        arrays = arrays.copy().reshape(16,8)
    if num_layers == 12:
        arrays= arrays.copy().reshape(16,16)
       
    cmap = colors.ListedColormap(['red','gray','white','black'])
    bounds=[-1, -1+1e-12, -1e-14, 1e-4 , 1]
    norm = colors.BoundaryNorm ( boundaries = bounds , ncolors = 4 )
    img = plt.imshow(arrays, interpolation='nearest', origin='lower',
                    cmap=cmap,norm=norm)

    plt.colorbar(img, cmap=cmap, boundaries=bounds,norm=norm)
    
    if save_pictures == True:
        title_obj = plt.title(name+str(num_layers)) #get the title property handler
        plt.getp(title_obj)                    #print out the properties of title
        plt.savefig('dissection/by_neuron/pictures/pdf/'+name+str(num_layers)+'.pdf')
    plt.show()
    

show_pictures(shift)    
#show_pictures(shift,'shift',True)    
show_pictures(var)

#print(var)    
#show_pictures(var,'var',True)    
    #%%
#%%
print(shift[0][2])

#%%
from myUtils import readData,saveData
import numpy as np 
def merge_zero_layer():
    for i in range(1,11):
        shift = readData("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p_value_by_neuron/p-value","p_val_shift_n"+str(i)+"_0")
        var  = readData("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p_value_by_neuron/p-value","p_val_variance_n"+str(i)+"_0")
        if i ==1:
            all_shift = shift
            all_var = var
        else:    
            all_shift = np.concatenate( (all_shift,shift),axis=2)
            all_var = np.concatenate( (all_var,var),axis=2)
    print(all_shift.shape)
    print(all_var.shape)
    saveData("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p_value_by_neuron/p-value","p_val_shift_n-1_0",all_shift)
    saveData("/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/p_value_by_neuron/p-value","p_val_variance_n-1_0",all_var)
#merge_zero_layer()

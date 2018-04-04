#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 09:35:22 2018

@author: andrey
"""

#%% work with dataset
import keras
path ='datasetsNN/landScapes/landScape_3000_32/Landscapes_3000_32x32_clear.hdf5'
X_data = keras.utils.io_utils.HDF5Matrix(path,'Landscapes')
print(X_data.shape)
#%%% генерация точек
import numpy
import random
import math
from neuralNetworks.landScape_Practica.IterativeAlgorithm import calculateResult



count_landscapes=3000
count_pair_points=15
count_shifts=8
count_pixels=32

def norm2(p1,p2):
    return math.sqrt(  (p2[0]-p1[0])*(p2[0]-p1[0])  +  (p2[1]-p1[1])*(p2[1]-p1[1])  )

def random_points():
    p1= (random.randint(0,count_pixels-1),random.randint(0,count_pixels-1))
    p2=p1
    while(norm2(p1,p2) < math.sqrt(8)):
        p2= (random.randint(0,count_pixels-1),random.randint(0,count_pixels-1))
    return p1,p2       


#point_data = numpy.zeros( (3000,15,8,32,32) )
point_data = numpy.zeros( ( count_landscapes, count_pair_points, count_shifts, count_pixels, count_pixels ) )
result=numpy.zeros( (count_landscapes, count_pair_points , count_shifts ))
#for el_15 in point_data:
#    for el_8 in el_15: # el_8 shape = (8,32,32)
#            for landscape in el_8:# shape= (32,32)
#                p1,p2=random_points()
#                landscape[p1] = 1
#                landscape[p2] = 1

for i in range(count_landscapes):
    print("START LANDSCAPE NUMBER:",i)
    for j in range(count_pair_points):
        for k in range(count_shifts):
            points_landscape=point_data[i][j][k]
            p1,p2=random_points()
            point_data[i][j][k][p1] = 1
            point_data[i][j][k][p2] = 1
            image= X_data[i][j][k]                                
            z1=image[p1]
            z2=image[p2]
            x1,y1,x2,y2 =p1[0],p1[1],p2[0],p2[1]
            if ( calculateResult( (x1,y1,z1),(x2,y2,z2),image ) == True ):
                result[i][j][k]=1
                    
            
#%%
print(result.shape)
import numpy as np
import h5py
h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/answers.hdf5', 'w')
h5f.create_dataset('answers_3000', data=result)
#%%
import numpy as np
import h5py
h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/points_landscape.hdf5', 'w')
h5f.create_dataset('answers_3000', data=point_data)
#%%

#%%
#import h5py
#import numpy as np
#f = h5py.File("datasetsNN/landscapes/landscape_3000_32/point.hdf5", "w")                
#dset = f.create_dataset("init", data=landscape) # че то не так




#%%
import numpy
data = numpy.zeros((3000,8,32,32))
for i in range(len(X_data)):
    data[i] = X_data[i][0]

print(data.shape) # (3000,8,32,32) 


#%%
#from neuralNetworks.landscape_Practica.IterativeAlgorithm import calculateResult
   
image = data[0][0]
p1 = numpy.zeros(2)
p2 = numpy.zeros(2)
p2=(2,1)
print(image.shape)    

x1=14
y1=2
p1 = (x1,y1)
z1=image[p1]

x2=10
y2=6
p2=(x2,y2)
z2=image[p2]
print("z",z1,z2)
print(calculateResult((x1,y1,z1),(x2,y2,z2),image))
print(image[12][3])


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:20:43 2018

@author: andrey
"""
#%%
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
def getModel(input_shape=(32,32,2)):
    model = Sequential()
    
    #1
    model.add(Conv2D(24,kernel_size=(3,3),padding='same',activation='relu',input_shape=input_shape)) # 32*32
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding='same')) #16*16
    model.add(BatchNormalization())

    
    #2
    model.add(Conv2D(64,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same'))  #8*8
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding='same')) #4*4
    model.add(BatchNormalization())

    
    #3
    model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu')) #4*4
    

    #4
    model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu')) #4*4
    
    
    #5
    model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu')) #4*4
    model.add(MaxPooling2D( pool_size=(3, 3), strides=(2,2), padding='same' )) #2*2

    model.add(Dropout(0.4)) # не уверен
    
    
    #fc layers    
    dense_size = 4*4*128 # то есть 2048
    model.add(Flatten()) #2048
    
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dense(1, activation='softmax'))  # че то совсем не уверен
    return model




#main 
model = getModel()
print(model.summary())


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


#%% work with dataset
import keras
path ='datasetsNN/landScapes/landScape_3000_32/Landscapes_3000_32x32_clear.hdf5'
X_data = keras.utils.io_utils.HDF5Matrix(path,'Landscapes')
print(X_data.shape)
#%%% генерация точек
import numpy
import random

def random_coord():
    return (random.randint(0,31),random.randint(0,31))       
point_data = numpy.zeros( (3000,15,8,32,32) )
print(random.randint(0,31))

for el_15 in point_data:
    for el_8 in el_15: # el_8 shape = (8,32,32)
            for landscape in el_8:
                landscape[random_coord()] = 1
#%%
import math                
def get_n(p1,p2): # единичный направляющий вектор
    v = diffVectors(p1,p2)
    norm = math.sqrt( v[0]*v[0]+v[1]*v[1] + v[2]*v[2] ) 
    return (v[0]/norm,v[1]/norm, v[2]/norm)               

def max(a,b):
    if (a>b):
        return a
    return b

def diffVectors(v1,v2):
    return (v2[0]-v1[0],v2[1]-v1[1],v2[2]-v1[2])

def multiNumber_Vector(a,v):
    return(a*v[0],a*v[1],a*v[2])

def residual(a,b):
    max_ab=  max( abs(a[0]-b[0]),abs(a[1]-b[1]))
    return max(max_ab,abs(a[2]-b[2]) )

def sumVector(v1,v2):
    return (v1[0]+v2[0],v1[1]+v2[1],v1[2]+v2[2])

def calculateResult(p1,p2, matrix):
    if ((p1[0]-p2[0]==0)and(p1[1]-p2[1]==0)and(p1[2]-p2[2]==0)):
        return True
    #flagValue = max( matrix[p1],matrix[p2] ) # если больше него то выходим
    tau=0.01 
    z = p1 # итеративная перменная.
    n = get_n(p1,p2)
    print('n=',n)
    n = multiNumber_Vector(tau,n)
    eps=0.0001
    countIter=0
    while( residual(z,p2) > eps):       
        countIter+=1
        if (countIter>1000):
            print("not work")
            return False
        z = sumVector(z,n)
        print("z",z[0],z[1])
        _,x =math.modf(z[0])
        _,y =math.modf(z[1])
        x = int(x)
        y = int(y)
        print("x,y",x,y)
        if ((x==p2[0])and(y==p2[1])):
            return True
        value = matrix[x][y]
        print("value",value)
        print("z[2]",z[2])
        if ( (z[2]<value) and  ((x!=p1[0])or(y!=p1[1])) ):
            print("countIter=",countIter)
            return False
        print(z[0],z[1],z[2])
        print("res=",residual(z,p2))
    print("countIter=",countIter)
    return True    
#%%
            
#%%
import h5py
import numpy as np
f = h5py.File("datasetsNN/landScapes/landScape_3000_32/point.hdf5", "w")                
dset = f.create_dataset("init", data=landscape) # че то не так
#%%
import numpy
data = numpy.zeros((3000,8,32,32))
for i in range(len(X_data)):
    data[i] = X_data[i][0]

print(data.shape) # (3000,8,32,32) 
#%%    
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

#%%

image1=numpy.random.randint(10,size=(3,3))
#%%
x1=0
y1=0
p1 = (x1,y1)
z1=image1[p1]
x2=2
y2=1
p2=(x2,y2)
z2=image1[p2]
print("z",z1,z2)
print(calculateResult((x1,y1,z1),(x2,y2,z2),image1))
print(image1)
print(image1[2][1])

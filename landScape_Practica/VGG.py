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
path ='datasetsNN/landScapes/landScape_3000_32/Landscapes_3000_32x32_clear.hdf5'
X_data = keras.utils.io_utils.HDF5Matrix(path,'Landscapes')
print(X_data.shape)

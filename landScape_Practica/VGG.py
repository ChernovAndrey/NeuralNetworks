#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:20:43 2018

@author: andrey
"""
#%%  import data
import keras
path ='datasetsNN/landScapes/landScape_3000_32/answers.hdf5'
result = keras.utils.io_utils.HDF5Matrix(path,'answers_3000')
print(result.shape)

path ='datasetsNN/landScapes/landScape_3000_32/points_landscape.hdf5'
points_landscapes= keras.utils.io_utils.HDF5Matrix(path,'answers_3000')
print(points_landscapes.shape)


path ='datasetsNN/landScapes/landScape_3000_32/Landscapes_3000_32x32_clear.hdf5'
landscapes_data = keras.utils.io_utils.HDF5Matrix(path,'Landscapes')
print(landscapes_data.shape)



#%%
import h5py
h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/answers.hdf5','r')
b = h5f['answer_3000'][:]
h5f.close()
#%% prepare data
import numpy as np

count_landscapes, count_pair_points, count_shifts, count_pixels= landscapes_data.shape[:4] 

data = np.zeros( shape=( count_landscapes, count_pair_points, count_shifts, 2,count_pixels, count_pixels ) )

for i in range(count_landscapes):
    for j in range(count_pair_points):
        for k in range(count_shifts):
           data[i][j][k][0] = landscapes_data[i][j][k]          
           data[i][j][k][1] = points_landscapes[i][j][k]          
           
print(data.shape)
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
#    dense_size = 2*2*128 # то есть 512
    model.add(Flatten()) #2048
    
    model.add(Dense(32, activation='relu')) #32
#    model.add(Dense(dense_size, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # че то совсем не уверен #sigmoid
    return model

#EPOCH 30


#main 
model = getModel()
print(model.summary())



model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

            
#%%


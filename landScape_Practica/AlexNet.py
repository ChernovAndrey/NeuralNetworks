#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:46:16 2018

@author: andrey
"""

#%%
work_dir ='/home/andrey/datasetsNN/landScapes/landScape_3000_32/mix_fixed/'
from myUtils import readData

x_train = readData(work_dir+'dataset_30_07_2018','x_train')
x_test = readData(work_dir+'dataset_30_07_2018','x_test')
y_train = readData(work_dir+'dataset_30_07_2018','y_train')
y_test  = readData(work_dir+'dataset_30_07_2018','y_test')
#%%
import keras
import math
import keras.initializers as ki
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

   
batch_size=256
epochs=100
def getModel(input_shape=(32,32,2)):
    model = Sequential()
    

    model.add(Conv2D(32,kernel_size=(5,5),padding='same',activation='relu',strides=(1,1),input_shape=input_shape)) # 32*32
    model.add(MaxPooling2D(pool_size=(2,2))) #16*16
    model.add(BatchNormalization())


    model.add(Conv2D(64,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same'))  #8*8
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    
    #3 64
    model.add(Conv2D(96,kernel_size=(3,3),activation='relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.001)))  #8*8
        
    #4 128
#    model.add(Dropout(0.15))
    model.add(Conv2D(96,kernel_size=(3,3),padding='same',activation='relu')) #4*4
  
    #5
    model.add(Dropout(0.4))
    model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu')) #4*4
    model.add(MaxPooling2D( pool_size=(2, 2))) #2*2
    model.add(BatchNormalization())

    
    
    #fc layers    
#    dense_size = 2*2*128 # то есть 512
    model.add(Flatten()) #2048
    
    model.add(Dense(128, activation='relu')) #32

#    model.add(Dense(8, activation='relu')) #8

#    model.add(Dense(dense_size, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # че то совсем не уверен #sigmoid
    return model

#EPOCH 30


#main 
model = getModel()
print(model.summary())

#%%
opt = keras.optimizers.nadam(lr=0.002)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model_AlexNet_second.hdf5')
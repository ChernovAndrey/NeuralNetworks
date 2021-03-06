#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:35:05 2018

@author: andrey
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:46:16 2018

@author: andrey
"""

# %%
from myUtils import getTestData2Points
import numpy as np
import h5py

name_X = 'dataset_11_09_2018'
#name_y = 'results_11_09_2018'
name_y = name_X

#name_model = 'model_AlexNet_fullConv_30_07_2018_2.hdf5'

h5f = h5py.File(name_X, 'r')
x_train = h5f['X_train'][...]
h5f.close()
# %%
print(x_train.shape)
# %%
h5f = h5py.File(name_y, 'r')
y_train = h5f['y_train'][...]
h5f.close()

h5f = h5py.File(name_X, 'r')
x_test = h5f['X_test'][...]
h5f.close()

h5f = h5py.File(name_y, 'r')
y_test = h5f['y_test'][...]
h5f.close()

print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)
print('x_test: ', x_test.shape)
print('y_test: ', y_test.shape)

# %%
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv1D, Reshape, Activation, GlobalAveragePooling2D
batch_size = 256
epochs = 100


def getModel(input_shape=(32, 32, 2)):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(7, 7), padding='same', activation='relu', strides=(1, 1),
                     input_shape=input_shape))  # 32*32
    model.add(MaxPooling2D(pool_size=(3, 3)))  # 16*16
    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))  # 8*8
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # 3 64
    #    model.add(Conv2D(96,kernel_size=(3,3),activation='relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.001)))  #8*8
    #
    #    #4 128
    ##    model.add(Dropout(0.15))
    #    model.add(Conv2D(96,kernel_size=(3,3),padding='same',activation='relu')) #4*4
    #
    #    #5
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))  # 4*4
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 2*2
    model.add(BatchNormalization())

    # fc layers
    #    dense_size = 2*2*128 # то есть 512
#    model.add(Flatten())  # 2048
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1))
#    model.add(Reshape((32)))
    #    model.add(Dense(32, activation='relu')) #32
#    model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))  # 32
#    model.add(Conv1D(filters=8, kernel_size=2, activation='relu'))  # 32
#    model.add(Conv1D(filters=1, kernel_size=2))  # 32
#    model.add(Flatten())
#    model.add(Activation('sigmoid'))
    return model


# EPOCH 30


# main
model = getModel()
print(model.summary())

# %%
opt = keras.optimizers.Nadam(lr=0.0002)
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

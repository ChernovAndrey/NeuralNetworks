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
import numpy as np
import h5py

h5f = h5py.File('mix/trainData.hdf5', 'r')
x_train = h5f['train_3000_'][...]
h5f.close()
# %%
print(x_train.shape)
# %%
h5f = h5py.File('mix/trainResult.hdf5', 'r')
y_train = h5f['result_3000'][...]
h5f.close()

h5f = h5py.File('mix/testData.hdf5', 'r')
x_test = h5f['test_3000_'][...]
h5f.close()

h5f = h5py.File('mix/testResult.hdf5', 'r')
y_test = h5f['resultTest_3000'][...]
h5f.close()

# %%
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv1D, Reshape, Activation
import keras.initializers as ki

batch_size = 256
epochs = 300
uniform_min_val = 1e-5
uniform_max_val = 1e-2


def getModel(input_shape=(32, 32, 2)):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(7, 7), padding='same', activation='relu', strides=(1, 1), input_shape=input_shape,
                     kernel_initializer=ki.RandomUniform(uniform_min_val, uniform_max_val)))  # 32*32
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(3, 3)))  # 16*16
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same',
                     kernel_initializer=ki.RandomUniform(uniform_min_val, uniform_max_val)))  # 8*8
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',
                     kernel_initializer=ki.RandomUniform(uniform_min_val, uniform_max_val)))  # 4*4
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 2*2
    model.add(BatchNormalization())

    # fc layers
    #    dense_size = 2*2*128 # то есть 512
    #   model.add(Flatten()) #2048
    model.add(Reshape((-1, 32)))
    model.add(Conv1D(32, 2, activation='relu'), kernel_initializer=ki.RandomUniform(uniform_min_val, uniform_max_val))
    model.add(Conv1D(8, 2, activation='relu'), kernel_initializer=ki.RandomUniform(uniform_min_val, uniform_max_val))
    model.add(Conv1D(1, 2), kernel_initializer=ki.RandomUniform(uniform_min_val, uniform_max_val))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    #    model.add(Dense(64, activation='relu')) #32
    #    model.add(Dense(32, activation='relu')) #32

    #    model.add(Dense(8, activation='relu')) #8

    #    model.add(Dense(dense_size, activation='relu'))
    #    model.add(Dense(1, activation='sigmoid'))  # че то совсем не уверен #sigmoid
    return model


# EPOCH 30


# main
model = getModel()
print(model.summary())

# %%
opt = keras.optimizers.Nadam(lr=0.002)
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

model.save('model_AlexNet_second_fullConv300.hdf5')

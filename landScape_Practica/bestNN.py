#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:46:16 2018

@author: andrey
"""

# %%
from myUtils import getTestData2Points

x_train, y_train, x_test, y_test = getTestData2Points()

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

batch_size = 256
epochs = 100


def getModel(input_shape=(32, 32, 2)):
    model = Sequential()

    # 1 24 3x3
    model.add(Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))  # 32*32
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))  # 16*16
    model.add(BatchNormalization())

    # 2 64
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))  # 8*8
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))  # 4*4
    model.add(BatchNormalization())

    # 3 128
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))  # 4*4

    model.add(Dropout(0.1))
    # 4
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))  # 4*4

    model.add(Dropout(0.2))

    # 5
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))  # 4*4
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))  # 2*2

    # fc layers
    #    dense_size = 2*2*128 # то есть 512
    model.add(Flatten())  # 2048

    model.add(Dense(32, activation='relu'))  # 32

    #    model.add(Dense(8, activation='relu')) #8

    #    model.add(Dense(dense_size, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # че то совсем не уверен #sigmoid
    return model


# EPOCH 30


# main
model = getModel()
print(model.summary())

# %%
model.compile(optimizer='nadam',
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

model.save('model_twoDropOut.hdf5')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:50:26 2018

@author: andrey
"""
#%%
from keras.layers import Input, Embedding, LSTM, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from keras.models import Model
import keras
import numpy as np
from myUtils import readData

def getIndexNonZeroInMatrix(matrix):
    nZero = np.nonzero(matrix)
    return nZero[0][0], nZero[1][0]   

def getPoints(data):
    Points=np.zeros( (len(data),6) ) 
    for i in range(len(data)):
        matrixes = np.reshape(data[i],(4,32,32))
        p=np.zeros((6))
        indp=0
        indm=1
        while(indp<5):
            p[indp],p[indp+1] = getIndexNonZeroInMatrix(matrixes[indm])
            indp+=2
            indm+=1
        Points[i]=p
    return Points    
#%%
x_train = readData('clearTrainData.hdf5','input')
x_test=readData('clearTestData.hdf5','input')

y_train =readData('clearTrainResult.hdf5','result')
y_test = readData('clearTestResult.hdf5','result')

p_train = getPoints(x_train)
p_test = getPoints(x_test)

x_train = x_train[:,:,:,:1] # 360000*32*32*4 ->360000*32*32*1
x_test = x_test[:,:,:,:1] # 360000*32*32*4 ->360000*32*32*1

print("x")
print(x_train.shape)
print(x_test.shape)
print("y")
print(y_train.shape)
print(y_test.shape)
print("points")
print(p_train.shape)
print(p_test.shape)

#%%
batch_size=256
epochs=100

main_input = Input(shape=(32,32,1), dtype='float32', name='main_input')

x = Conv2D( 64,kernel_size=(5,5),padding='same',activation='relu',strides=(1,1) )(main_input) 
x = MaxPooling2D(pool_size=(2,2))(x) #16*16
x = BatchNormalization()(x)

x = Conv2D(96,kernel_size=(3,3),strides=(2,2),activation='relu',padding='same')(x) 
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)


#3 64
x = Conv2D(96,kernel_size=(3,3),activation='relu',padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(x)  #8*8
        
    #4 128
x = Conv2D(96,kernel_size=(3,3),padding='same',activation='relu')(x) #4*4

x = Dropout(0.2)(x) 
x = Conv2D(64,kernel_size=(3,3),padding='same',activation='relu')(x) #4*4
x = MaxPooling2D( pool_size=(2, 2))(x) #2*2
x = BatchNormalization()(x)
x = Flatten()(x)

x = Dense(252, activation='relu')(x) #32
 
points_input = Input(shape=(6,), name='points_input')

x = keras.layers.concatenate([x, points_input])

x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)

main_output = Dense(3, activation='sigmoid', name='main_output')(x)


model = Model(inputs=[main_input, points_input], outputs=[main_output])

print(model.summary()) 
#%%
opt = keras.optimizers.Nadam(lr=0.0002)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit([x_train,p_train], y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test))

score = model.evaluate([x_test,p_test], y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model_AlexNet_Graph_3Points.hdf5')




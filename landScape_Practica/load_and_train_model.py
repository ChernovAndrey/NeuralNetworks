#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:39:09 2018

@author: andrey
"""
#%%
name_model = ''
import keras
from keras.models import load_model
from myUtils import getTestData2Points
x_train,y_train,x_test,y_test = getTestData2Points()


batch_size=256
epochs=25
model = load_model(name_model)
print(model.summary())

opt = keras.optimizers.nadam(lr=0.001)
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

model.save('model_Alexnet_full_conv_325.hdf5')



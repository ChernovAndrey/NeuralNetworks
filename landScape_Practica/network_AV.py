#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:39:09 2018

@author: andrey
"""
#%%
from keras.models import load_model



import h5py
h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/mix/trainData.hdf5','r')
x_train = h5f['train_3000_'][...]
h5f.close()


h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/mix/trainResult.hdf5','r')
y_train = h5f['result_3000'][...]
h5f.close()


h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/mix/testData.hdf5','r')
x_test = h5f['test_3000_'][...]
h5f.close()



h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/mix/testResult.hdf5','r')
y_test = h5f['resultTest_3000'][...]
h5f.close()


batch_size=256
epochs=100
model = load_model('datasetsNN/landScapes/landScape_3000_32/weights_AV/weights.hdf5')
print(model.summary())

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

model.save('datasetsNN/landScapes/landScape_3000_32/mix/weights.hdf5')



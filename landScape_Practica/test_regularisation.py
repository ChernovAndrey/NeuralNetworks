#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:12:20 2018

@author: andrey
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:35:05 2018

@author: andrey
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:46:16 2018

@author: andrey
"""
#%%
#%%
import numpy as np
from keras.models import load_model
import h5py

name_model = 'model_AlexNet_fullConv_30_07_2018_2.hdf5'

h5f = h5py.File('dataset_30_07_2018','r')
x_train = h5f['x_train'][...]
h5f.close()
#%%
print(x_train.shape)
#%%
h5f = h5py.File('dataset_30_07_2018','r')
y_train = h5f['y_train'][...]
h5f.close()


h5f = h5py.File('dataset_30_07_2018','r')
x_test = h5f['x_test'][...]
h5f.close()



h5f = h5py.File('dataset_30_07_2018','r')
y_test = h5f['y_test'][...]
h5f.close()

#%%
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv1D, Reshape, Activation
from keras.layers import GlobalAveragePooling2D

batch_size=256
epochs=100

def getModel1(input_shape=(32,32,2)):
    model = Sequential()
    

    model.add(Conv2D(64,kernel_size=(7,7),padding='same',activation='relu',strides=(1,1),input_shape=input_shape)) # 32*32
   # model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3,3))) #16*16
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    
    model.add(Conv2D(64,kernel_size=(5,5),strides=(1,1),activation='relu',padding='same'))  #8*8
   # model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu')) #4*4
    model.add(MaxPooling2D( pool_size=(2, 2))) #2*2
    model.add(BatchNormalization())

    
    
    model.add(Reshape((-1,32)))
    model.add(Conv1D(32,2,activation='relu'))
    model.add(Conv1D(8,2,activation='relu'))
    model.add(Conv1D(1,2))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    return model

def getModel2(input_shape=(32,32,2)):
    model = Sequential()
    

    model.add(Conv2D(64,kernel_size=(7,7),padding='same',activation='relu',strides=(1,1),input_shape=input_shape)) # 32*32
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(3,3))) #16*16
    model.add(BatchNormalization())
#    model.add(Dropout(0.2))
  
    
    model.add(Conv2D(64,kernel_size=(5,5),strides=(1,1),activation='relu',padding='same'))  #8*8
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
#    model.add(Dropout(0.2))
    
    
    
    model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu')) #4*4
    model.add(MaxPooling2D( pool_size=(2, 2))) #2*2
    model.add(BatchNormalization())

    
    model.add(Reshape((-1,32)))
    model.add(Conv1D(32,2,activation='relu'))
    model.add(Conv1D(8,2,activation='relu'))
    model.add(Conv1D(1,2))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    return model

def getModel3(input_shape=(32,32,2)):
    model = Sequential()
    

    model.add(Conv2D(64,kernel_size=(7,7),padding='same',activation='relu',strides=(1,1),input_shape=input_shape)) # 32*32
    model.add(MaxPooling2D(pool_size=(3,3))) #16*16
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
#    model.add(Dropout(0.2))
  
    
    model.add(Conv2D(64,kernel_size=(5,5),strides=(1,1),activation='relu',padding='same'))  #8*8
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
#    model.add(Dropout(0.2))
    
    
    
    model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu')) #4*4
    model.add(MaxPooling2D( pool_size=(2, 2))) #2*2
    model.add(BatchNormalization())

    
    model.add(Reshape((-1,32)))
    model.add(Conv1D(32,2,activation='relu'))
    model.add(Conv1D(8,2,activation='relu'))
    model.add(Conv1D(1,2))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    return model

def getModel4(input_shape=(32,32,2)):
    model = Sequential()
    

    model.add(Conv2D(64,kernel_size=(7,7),padding='same',activation='relu',strides=(1,1),input_shape=input_shape)) # 32*32
    model.add(MaxPooling2D(pool_size=(3,3))) #16*16
#    model.add(Dropout(0.2))
#    model.add(BatchNormalization())
#    model.add(Dropout(0.2))
  
    
    model.add(Conv2D(64,kernel_size=(5,5),strides=(1,1),activation='relu',padding='same'))  #8*8
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.2))
#    model.add(BatchNormalization())
#    model.add(Dropout(0.2))
    
    
    
    model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu')) #4*4
    model.add(MaxPooling2D( pool_size=(2, 2))) #2*2
#    model.add(BatchNormalization())
    
    
    model.add(Reshape((-1,32)))
    model.add(Conv1D(32,2,activation='relu'))
    model.add(Conv1D(8,2,activation='relu'))
    model.add(Conv1D(1,2))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    return model

def getModel5(input_shape=(32,32,2)):
    model = Sequential()
    

    model.add(Conv2D(64,kernel_size=(7,7),padding='same',activation='relu',strides=(1,1),input_shape=input_shape)) # 32*32
    model.add(MaxPooling2D(pool_size=(3,3))) #16*16
    model.add(Dropout(0.2))
#    model.add(BatchNormalization())
#    model.add(Dropout(0.2))
  
    
    model.add(Conv2D(64,kernel_size=(5,5),strides=(1,1),activation='relu',padding='same'))  #8*8
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
#    model.add(BatchNormalization())
#    model.add(Dropout(0.2))
    
    
    
    model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu')) #4*4
    model.add(MaxPooling2D( pool_size=(2, 2))) #2*2
#    model.add(BatchNormalization())
    
    
    model.add(Reshape((-1,32)))
    model.add(Conv1D(32,2,activation='relu'))
    model.add(Conv1D(8,2,activation='relu'))
    model.add(Conv1D(1,2))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    return model

def getModel6(input_shape=(32,32,2)):
    model = Sequential()
    

    model.add(Conv2D(64,kernel_size=(7,7),padding='same',activation='relu',strides=(1,1),input_shape=input_shape)) # 32*32
   # model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3,3))) #16*16
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    
    model.add(Conv2D(64,kernel_size=(5,5),strides=(1,1),activation='relu',padding='same'))  #8*8
   # model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu')) #4*4
    model.add(MaxPooling2D( pool_size=(2, 2))) #2*2
    model.add(BatchNormalization())
    
    model.add(Conv2D(16,kernel_size=(3,3),padding='same',activation='relu')) #4*4
    model.add(Conv2D(1,kernel_size=(3,3),padding='same',activation='relu')) #4*4

    model.add( GlobalAveragePooling2D() )

    
#    model.add(Reshape((-1,32)))
#    model.add(Reshape())
#    model.add(Flatten())
#    model.add(Conv1D(16,2,activation='relu', padding='same'))
#    model.add(Conv1D(8,2,activation='relu', padding='same'))
#    model.add(Conv1D(1,2,padding='same'))
#    model.add( GlobalAveragePooling2D() )
#    model.add(Flatten())
    model.add(Activation('sigmoid'))
    return model




##main 
#model = getModel6()
##model = load_model(name_model)
#print(model.summary())


#%%
def learning(model,x_train,y_train,x_test,y_test,epochs,batch_size,nModel):
    opt = keras.optimizers.Nadam(lr=0.0001)
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
    model.save('model_AlexNet_fullConv_test_'+str(nModel)+'.hdf5')
#%%
model = getModel1()
print(model.summary())
learning(model,x_train,y_train,x_test,y_test,epochs,batch_size,1)

model = getModel2()
print(model.summary())
learning(model,x_train,y_train,x_test,y_test,epochs,batch_size,2)

model = getModel3()
print(model.summary())
learning(model,x_train,y_train,x_test,y_test,epochs,batch_size,3)

model = getModel4()
print(model.summary())
learning(model,x_train,y_train,x_test,y_test,epochs,batch_size,4)

model = getModel5()
print(model.summary())
learning(model,x_train,y_train,x_test,y_test,epochs,batch_size,5)

model = getModel6()
print(model.summary())
learning(model,x_train,y_train,x_test,y_test,epochs,batch_size,6)

#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:20:43 2018

@author: andrey
"""
#%%  import data
import keras


path ='datasetsNN/landScapes/landScape_3000_32/Landscapes_3000_32x32_clear.hdf5'
landscapes_data = keras.utils.io_utils.HDF5Matrix(path,'Landscapes')
print(landscapes_data.shape)



#%%
#import h5py
#h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/answers.hdf5','r')
#result = h5f['answers_3001'][...]
#h5f.close()
#
#
#h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/points_landscape.hdf5','r')
#points_landscapes = h5f['points_3000'][...]
#h5f.close()



import numpy as np
import h5py
h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/2/points.hdf5','r')
points_landscapes = h5f['points_3000_2'][...]
h5f.close()


h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/2/result.hdf5','r')
result = h5f['resultss_3000_2'][...]
h5f.close()
#%% prepare data
print(result.shape)
print(points_landscapes.shape)
#%%
import numpy as np

count_landscapes, count_pair_points, count_shifts, count_pixels= landscapes_data.shape[:4] 

data = np.zeros( shape=( count_landscapes, count_pair_points, count_shifts, 2,count_pixels, count_pixels ), dtype = np.float16 )
print(data.shape)
print(landscapes_data.shape)
print(points_landscapes.shape)
for i in range(count_landscapes):
    for j in range(count_pair_points):
        for k in range(count_shifts):
           data[i][j][k][0] = landscapes_data[i][j][k]          
           data[i][j][k][1] = points_landscapes[i][j][k]          
           
print(data.shape)
first_dim=count_landscapes*count_pair_points*count_shifts
data = np.reshape(data,(first_dim,2,count_pixels,count_pixels))

#%%
print(data.shape)

#%%%
firstDim= data.shape[0]

data = data.reshape(firstDim,32,32,2)
print(data.shape)
#%%
import numpy as np
import h5py
h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/2/ready_data.hdf5', 'w')
h5f.create_dataset('dataset_3000', data=data,dtype=np.float16)
h5f.close()
#%%
result=result.reshape(-1)
#%%
print(result.shape)
#%%

import numpy as np
import h5py
h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/2/ready_res.hdf5', 'w')
h5f.create_dataset('dataset_3000', data=result,dtype=np.float32)
h5f.close()


#%%
import numpy as np
import h5py
h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/2/ready_data.hdf5','r')
data = h5f['dataset_3000'][...]
h5f.close()


h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/2/ready_res.hdf5','r')
result = h5f['dataset_3000'][...]
h5f.close()

print(data.shape)
print(result.shape)

count_train=300000
x_train=data[:count_train]
x_test=data[count_train:]
y_train=result[:count_train]
y_test=result[count_train:]
#%%
print(y_test[100:120])
#%%
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D


batch_size=256
epochs=30

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


#%%
model.compile(optimizer='nadam',
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

model.save('datasetsNN/landScapes/landScape_3000_32/weights.hdf5')
            
#%%
from keras.models import load_model

model = load_model('datasetsNN/landScapes/landScape_3000_32/weights.hdf5')

#%%

res_predict = model.predict(x_test)

#%%
import numpy as np
def saveData(path,name,data):
    import h5py
    h5f = h5py.File(path, 'w')
    h5f.create_dataset(name, data=data,dtype=np.float32)
    h5f.close()
#%%
import numpy as np    
from sklearn.metrics import confusion_matrix
def getInt(vector): # zero or one
    res=np.zeros(vector.shape, dtype=int)
    for i in range(len(vector)):
        if(vector[i]>=0.5):
            res[i] = 1
        else:
            res[i]=0
    return res

y_pred= res_predict.reshape(-1)

res_pred=getInt(y_pred)
res_test=getInt(y_test)


y_test.sum()
#print(res_pred[20:30])
#print(res_test[20:30])

confusion_matrix(res_test,res_pred)
    
#%%
from sklearn.metrics import precision_score,recall_score,f1_score
print("Точность:",precision_score(res_test,res_pred))
print("Полнота:",recall_score(res_test,res_pred))
print("F1:",f1_score(res_test,res_pred))


#%%
print(y_test[100:110])
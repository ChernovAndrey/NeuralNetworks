#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:30:34 2018

@author: andrey
"""

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
from keras.models import load_model

model = load_model('datasetsNN/landScapes/landScape_3000_32/weights_AV/weights.hdf5')
print(model.summary())
#%%

from sklearn.metrics import confusion_matrix

res_predict = model.predict(x_test)

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

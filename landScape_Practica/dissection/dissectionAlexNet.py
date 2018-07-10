#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 17:12:33 2018

@author: andrey
"""
countSamples = 1000
#%%
from landScape_Practica.myUtils import getTestData2Points
x_test,y_test = getTestData2Points()

#%%
from keras.models import load_model

#print(keras.__version__)
#model =  keras.models.load_model('model_AlexNet_Graph_3PointsClear.hdf5')
model = load_model('/home/andrey/datasetsNN/landScapes/landScape_3000_32/AlexNet/model_AlexNet_second.hdf5')
print(model.summary())
#print(model.summary())
#%%
print(x_test.shape)
#%%

from landScape_Practica.keras_utils import get_layer_output

output = get_layer_output(model,x_test[:countSamples])
#%%
print(len(output))
print(output[0].shape)
#%%
print(output[13].shape)
#%%
import numpy as np
import h5py 
def getInd2Class(y_test): #разбиваем x_test на два класса, первый где y_test=0, второй, где y_test=1, возвращаем индексы
    ind0 = np.where(y_test==0)[0]
    ind1 = np.where(y_test==1)[0]
    return ind0,ind1
                
def saveDistribution(vector,step,flagRes,numLayers): # сохраняем numpy histogram; flagRes=0, если res0; если res1, то 1.
    bins = []
    i=0.0
    print("max=",max(vector))
    maxV=max(vector)
    while i<maxV+1e-2:
        bins.append(i)
        i+=step
    plot0=np.histogram(vector,bins)
    x = plot0[1]
    y = plot0[0]
    x_aver = np.zeros(y.shape)
    for i in range(len(y)):
        x_aver[i] = (x[i]+x[i+1])/2.0
   # plt.plot(x_aver,y)
    h5f = h5py.File('outLayers1', 'a')
    h5f.create_dataset('x'+str(numLayers)+"_"+str(flagRes), data=x_aver,dtype=np.float32)
    h5f.create_dataset('y'+str(numLayers)+"_"+str(flagRes), data=y,dtype=np.float32)
    h5f.close()
    return x_aver,y
#%% для последнего
#    ind0,ind1 = getInd2Class(y_test[:countSamples])
#    showDistribution(output[14][ind0],0.05)
#    showDistribution(output[14][ind1],0.05)
#%%
#showDistribution(output[14],0.1)
    
#%%
    
print(len(model.layers))
ind0,ind1 = getInd2Class(y_test[:countSamples])
for i in range(len(model.layers)):
    res0 = output[i][ind0] #где y_tes=0
    res1 = output[i][ind1]#где y_test=1

    h5f = h5py.File('result', 'a')
    h5f.create_dataset('res'+str(i)+"_0", data=res0,dtype=np.float32)
    h5f.create_dataset('res'+str(i)+"_0", data=res1,dtype=np.float32)
    h5f.close()

    #saveDistribution(res0,max(res0)/2.0, 0,i)
    #saveDistribution(res1,max(res1)/2.0, 1,i)
    #plt.show()    
#%%


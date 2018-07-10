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
from landScape_Practica.myUtils import saveData
    
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
#%% generate random samples for check hypot
from landScape_Practica.myUtils import readData,saveData
import numpy as np
def gen_rand_samples_for_hypot(num_layers,num_repeat=100, num_samples = 5000):
    res0=np.empty(shape=0,dtype=int)
    res1=np.empty(shape=0,dtype=int)

    allRes0 = readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/allResult',
                         'res'+str(num_layers)+'_'+'0') 
    allRes1 = readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/allResult',
                         'res'+str(num_layers)+'_'+'1') 
               
    half_num_samples = num_samples//2
    for i in range(num_repeat):
        rand_num0 = np.random.randint(allRes0.shape[0], size=half_num_samples)     # 60000- всего экземпляров
        rand_num1 = np.random.randint(allRes1.shape[0], size=half_num_samples)     # 60000- всего экземпляров
        for j in range(half_num_samples):
                k0 = rand_num0[j]
                k1 = rand_num1[j]
                res0=np.concatenate( ( res0,allRes0[k0].reshape(-1) ) )
                res1=np.concatenate( ( res1, allRes1[k1].reshape(-1) ) )

    print(res0.shape)
    print(res1.shape)
    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res'+str(num_layers)+'_0',res0)
    saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/rand_5000_samples_result','res'+str(num_layers)+'_1',res1)
    


#%%
gen_rand_samples_for_hypot(14,100,5000) 
    
#%%
import numpy as np
res=np.empty(shape=0,dtype=int)
num_layers=14
for i in range(1,7):
    res=np.concatenate( ( res,(readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/result'+str(i),
                                          'res'+str(num_layers)+'_'+'1')).reshape(-1) ) )
print(res.shape)    
#%%%
saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/dissection/allResult','res'+str(num_layers)+'_1',res)


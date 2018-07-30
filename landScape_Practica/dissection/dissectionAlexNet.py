#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 17:12:33 2018

@author: andrey
"""
#%%
from landScape_Practica.myUtils import getTestData2Points
x_test,y_test = getTestData2Points()
#%%
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

output = get_layer_output(model,x_test)
#%%
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
ind0,ind1 = getInd2Class(y_test)
for i in range(len(model.layers)):
    res0 = output[i][ind0] #где y_tes=0
    res1 = output[i][ind1]#где y_test=1

    h5f = h5py.File('result', 'a')
    h5f.create_dataset('res'+str(i)+"_0", data=res0,dtype=np.float32)
    h5f.create_dataset('res'+str(i)+"_1", data=res1,dtype=np.float32)
    h5f.close()

    #saveDistribution(res0,max(res0)/2.0, 0,i)
    #saveDistribution(res1,max(res1)/2.0, 1,i)
    #plt.show()    
#%%
from keras.models import load_model
from myUtils import readData
from landScape_Practica.keras_utils import get_layer_output

#сохраняет разделенные(по какому-либо признаку) результаты для нужных слоев 
#path_x_test - список из двух строк. Первая строка название файла, вторая название датасета.
#n_layers - номера слоев, которые нужно исследовать
#f_divided  - функция, которая делит данные( по меткам). Возращает индексы двух групп.
def save_divided_output_layers(work_dir,name_model,path_x_test,path_y_test,n_layers='all',f_divided):
    model = load_model(work_dir+name_model)
    x_test =readData(*path_x_test)
    y_test =readData(*path_y_test)
    #log
    output = get_layer_output(model,x_test,n_layers)
    #log
    ind0, ind1 = f_divided(y_test)
    if n_layers = 'all':
        n_layers = range(0,len(model.layers))
    file_save = 'out_layers'
    for i in n_layers:
        res0 = output[i][ind0] #где y_tes=0
        res1 = output[i][ind1]#где y_test=1
        saveData(file_save,'out'+str(i)_'0',res0)        
        saveData(file_save,'out'+str(i)_'1',res1)
    #log        
#%% 
  
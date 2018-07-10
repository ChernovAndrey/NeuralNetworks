#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 18:30:39 2018

@author: andrey
"""
#%%
import numpy as np

def saveData(path,name,data):
    h5f = h5py.File(path, 'w')
    h5f.create_dataset(name, data=data,dtype=np.float32)

def readData(path,name):
    import h5py
    h5f = h5py.File(path,'r')
    result = h5f[name][...]
    h5f.close()
    return result
#%%
dp1 =readData('clearDataset.hdf5','input')    
res1= readData('clearResult.hdf5','result')    
rej1 = readData('clearReject.hdf5','reject') 
#%%
dp2 =readData('clearDatasetOnes.hdf5','input')    
res2= readData('clearResultOnes.hdf5','result')    
rej2 = readData('clearRejectOnes.hdf5','reject') 
#%%
countTrain1=0.9*dp1.shape[0]
countTrain2=0.9*dp2.shape[0]

dp1Train=dp1[:countTrain1]
res1Train=res1[:countTrain1]
rej1Train=rej1[:countTrain1]

dp2Train=dp2[:countTrain2]
res2Train=res2[:countTrain2]
rej2Train=rej2[:countTrain2]


# test 
dp1Test=dp1[countTrain1:]
res1Test=res1[countTrain1:]
rej1Test=rej1[countTrain1:]

dp2Test=dp2[countTrain2:]
res2Test=res2[countTrain2:]
rej2Test=rej2[countTrain2:]

#%%%
all_dpTrain = [dp1Train,dp2Train]
all_resTrain = [res1Train,res2Train]
all_rejTrain = [rej1Train,rej2Train]

all_dpTrain=np.stack(all_dpTrain, axis=0)
all_resTrain=np.stack(all_resTrain, axis=0)
all_rejTrain=np.stack(all_rejTrain, axis=0)

print("Train shape")
print("dp",all_dpTrain.shape)
print("res",all_resTrain.shape)
print("rej",all_rejTrain.shape)
#%%
all_dpTest = [dp1Test,dp2Test]
all_resTest = [res1Test,res2Test]
all_rejTest = [rej1Test,rej2Test]

all_dpTest=np.stack(all_dpTest, axis=0)
all_resTest=np.stack(all_resTest, axis=0)
all_rejTest=np.stack(all_rejTest, axis=0)

print("Test shape")
print("dp",all_dpTest.shape)
print("res",all_resTest.shape)
print("rej",all_rejTest.shape)
#%%



#


#%%
def unison_shuffled_copies(a, b,c):
    assert len(a) == len(b)
    assert len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

all_dpTrain,all_resTrain,all_rejTrain = unison_shuffled_copies(all_dpTrain,all_resTrain,all_rejTrain)

print("Train shape")
print("dp",all_dpTrain.shape)
print("res",all_resTrain.shape)
print("rej",all_rejTrain.shape)

#%%
#
#saveData('finishDSTrain.hdf5','input',all_dpTrain)    
#saveData('finishResultTrain.hdf5','result',all_resTrain)    
#saveData('finishRejectTrain.hdf5','reject',all_rejTrain)    
#
#saveData('finishDSTest.hdf5','input',all_dpTest)    
#saveData('clearResultTest.hdf5','result',all_resTest)    
#saveData('clearRejectTest.hdf5','reject',all_rejTest)    



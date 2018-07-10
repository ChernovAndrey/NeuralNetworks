#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:46:09 2018

@author: andrey
"""

#%%
import numpy as np
import h5py

def saveData(path,name,data):
    h5f = h5py.File(path, 'w')
    h5f.create_dataset(name, data=data,dtype=np.float32)
   
def readData(path,name):
    h5f = h5py.File(path,'r')
    result = h5f[name][...]
    h5f.close()
    return result
#%%stack part
#dp =readData('dataset.hdf5','input')    
#res= readData('result.hdf5','result')    
#rej = readData('reject.hdf5','reject')    
#
#
#dp2 = readData('dataset2.hdf5','input')    
#res2 = readData('result2.hdf5','result')    
#rej2 = readData('reject2.hdf5','reject')    
#
#
#all_dp = [dp,dp2]
#all_res = [res,res2]
#all_rej = [rej,rej2]
#
#all_dp=np.stack(all_dp, axis=1)
#all_res=np.stack(all_res, axis=1)
#all_rej=np.stack(all_rej, axis=1)
#%% if only fix

all_dp =readData('dataset.hdf5','input')    
all_res= readData('result.hdf5','result')    
all_rej = readData('reject.hdf5','reject')     

#%%
print(all_dp.shape)
print(all_res.shape)
print(all_rej.shape)

#countLand=all_dp.shape[0]
#countPair=all_dp.shape[1]
#countShift=all_dp.shape[2]
#      
#all_dp=all_dp.reshape(countLand*countPair*countShift,4,32,32)
#all_res=all_res.reshape(countLand*countPair*countShift,3)
#all_rej=all_rej.reshape(countLand*countPair*countShift,3)


bugEl = np.where(( (all_rej<0) & (all_res==1)))[0]
bugEl=np.unique(bugEl)
print("bugEl",bugEl.shape)

ind_clear= np.arange(countLand*countPair*countShift)

ind_clear = np.delete(ind_clear,bugEl)
print("ind_clear",ind_clear.shape)

clData=np.empty( (len(ind_clear),4,32,32) )
clRes=np.empty( (len(ind_clear),3) )
clRej=np.empty( (len(ind_clear),3) )

j=0
for i in range(len(ind_clear)):
    clData[j]=all_dp[ind_clear[i]]
    clRes[j]=all_res[ind_clear[i]]
    clRej[j]=all_rej[ind_clear[i]]
    j+=1


bugElFin = np.where(( (clRej<0) & (clRes==1)))[0]
print("bug fin",bugElFin.shape)
print(clData.shape)
print(clRes.shape)
print(clRej.shape)
#%%
#saveData('clearDataset.hdf5','input',clData)    
#saveData('clearResult.hdf5','result',clRes)    
#saveData('clearReject.hdf5','reject',clRej)    
 


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:40:15 2018

@author: andrey
"""
#%%
def readData(path,name):
    import h5py
    h5f = h5py.File(path,'r')
    result = h5f[name][...]
    h5f.close()
    return result
#%%
import numpy as np    
data=readData('ready3PointTestData.hdf5','testData')
result = readData('ready3PointTestResult.hdf5','testResult')
reject=readData('reject.hdf5','reject')
reject=reject.reshape( (reject.shape[0]*reject.shape[1]*reject.shape[2],-1) )[300000:]
print(reject.shape)

#%%
print(reject.shape)
print(result.shape)
#%%
ind_clear = np.where( ((reject<0) & (result==1)) )[0]
#ind_clear = np.where( np.logical_not( np.logical_and( (np.less(reject,0)),( np.equal(result,1) ) ) ) )[0] #  то есть res_test = 0 , а res_pred = 1
print(ind_clear[:10])
ind_clear=np.unique(ind_clear)
print(len(ind_clear))
print(ind_clear)
#%%
#
clData=np.empty( (len(ind_clear),data.shape[1],data.shape[2],data.shape[3]) )
clResult=np.empty( (len(ind_clear),result.shape[1]) )
clReject=np.empty( (len(ind_clear),reject.shape[1]) )


j=0
for i in range(len(ind_clear)):
    clData[j]=data[ind_clear[i]]
    clResult[j]=result[ind_clear[i]]
    clReject[j]=reject[ind_clear[i]]
    j+=1
print(clResult.shape)    
print(clReject.shape)    
#%%
clResult = clResult.reshape(-1)    
clReject = clReject.reshape(-1)

bugEl = np.where(( (clReject<0) & (clResult==1)))[0]
print(bugEl.shape)    
#print(clData.shape)
#print(clResult.shape)
#print(clReject.shape)

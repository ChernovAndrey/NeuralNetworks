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
#x_train=data[:count_train]
x_test=data[count_train:]
#y_train=result[:count_train]
y_test=result[count_train:]
#%%
from keras.models import load_model

model = load_model('datasetsNN/landScapes/landScape_3000_32/mix/model_twoDropOut.hdf5')

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
#%%
confusion_matrix(res_test,res_pred)
#%%
from sklearn.metrics import precision_score,recall_score,f1_score
print("Точность:",precision_score(res_test,res_pred))
print("Полнота:",recall_score(res_test,res_pred))
print("F1:",f1_score(res_test,res_pred))

#%%

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


def ROC_plot():
#    fpr = dict()
#    tpr = dict()
#    roc_auc = dict()
    fpr, tpr, thr = roc_curve(res_test, y_pred)
    roc_auc= auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    print(fpr)
    print(tpr)
    print(thr)
    
    print(res_test.shape)
    print(res_pred.shape)
ROC_plot()

#%%
print(np.sum(res_test))
print(res_test)
#%%
print(y_pred)

#%%
print(res_test)
print(res_pred)
#%%

a=np.arange(9)

np.where(a<4)
ind_FN,i)
    
#%%

ind_FP = np.where(( (res_test==0) & (res_pred==1)))[0] #  то есть res_test = 0 , а res_pred = 1
ind_FN = np.where(( (res_test==1) & (res_pred==0)))[0] #  то есть res_test = 1 , а res_pred = 0
print(ind_FP.shape)
print(ind_FN.shape)
#%%
import numpy as np
import math
print(x_test.shape)


def getPoints(matrix): #32*32
    p=np.zeros(shape=(2,2))
    k=0 # индекс по массиву p
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] ==1:
                p[k]=(i,j)
                k+=1
    return p                

def getDistance(p1,p2):
    return math.sqrt( (p1[0]-p2[0])*(p1[0]-p2[0])  + (p1[1]-p2[1])*(p1[1]-p2[1]) ) 
def getDistanceARR(ind, matrixData):
    dist = np.empty(shape=(len(ind)))
    print(dist.shape)
    for i in range(len(ind)):
        p = getPoints(matrixData[ind[i]][1])
        dist[i] = getDistance(p[0],p[1])
    return dist    
#        dist[i] = getDistance(p[0],p[1])

#matrixData = x_test.reshape( (len(x_test),2,32,32) )                
matrixData = x_test.reshape(60000,2,32,32)                
print(matrixData.shape)
distFP = getDistanceARR(ind_FP, matrixData)
distFN = getDistanceARR(ind_FN, matrixData)
print(dist.shape)
#%%

plt.hist(distFP, bins = [0,5,10,15,20,25,30,32]) 
plt.title("histogram") 
plt.show()
#%%

plt.hist(distFN, bins = [0,5,10,15,20,25,30,32]) 
plt.title("histogram") 
plt.show()
#%%
ind_all=np.arange(60000)
distAll=getDistanceARR(ind_all,matrixData)
#%% 

plt.hist(distAll, bins = [0,5,10,15,20,25,30,32]) 
plt.title("histogram") 
plt.show()
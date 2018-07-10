#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:30:34 2018

@author: andrey
"""
#%%
result=readData('testResult.hdf5','resultTest_3000')
result=result.reshape(-1)
print(result.shape)
print(len(result))
print(result.sum())
#print(result[10:50])
#%%
resultOnes=readData('clearResultOnes.hdf5','result')
resultOnes=resultOnes.reshape(-1)
print(resultOnes.shape)
print(len(resultOnes))
print(resultOnes.sum())


#%%
from landScape_Practica.myUtils import saveData

#%%
from landScape_Practica.myUtils import getTestData2Points
x_test,y_test = getTestData2Points()    
#%%
import numpy as np
import h5py

def readData(path,name):
    h5f = h5py.File(path,'r')
    result = h5f[name][...]
    h5f.close()
    return result
#%%  
y_test=readData('testResult.hdf5','resultTest_3000')

x_test = readData('testData.hdf5','test_3000_')
#%%
print(x_test.shape)
print(y_test.shape)
#%%
from keras.models import load_model
#print(keras.__version__)
#model =  keras.models.load_model('model_AlexNet_Graph_3PointsClear.hdf5')
model = load_model('/home/andrey/datasetsNN/landScapes/landScape_3000_32/AlexNet/model_AlexNet_second.hdf5')
print(model.summary())
#print(model.summary())
#%% только для graph cnn
p_test = getPoints(x_test)

x_test = x_test[:,:,:,:1] # 360000*32*32*4 ->360000*32*32*1
#%%
from sklearn.metrics import confusion_matrix

res_predict = model.predict([x_test,])
#%%
print(res_predict)
#%%
res_predict = np.reshape(res_predict,(-1))
y_test = np.reshape(y_test,(-1))
print(res_predict.shape)
print(y_test.shape)
#%%
import numpy as np
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
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
print("val accuracy",accuracy_score(res_test,res_pred))
print("Точность:",precision_score(res_test,res_pred))
print("Полнота:",recall_score(res_test,res_pred))
print("F1:",f1_score(res_test,res_pred)) # 0.856 лучшая; для 3 точек лучшая 0.742

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

ind_FP = np.where(( (res_test==0) & (res_pred==1)))[0] #  то есть res_test = 0 , а res_pred = 1
ind_FN = np.where(( (res_test==1) & (res_pred==0)))[0] #  то есть res_test = 1 , а res_pred = 0
print(ind_FP.shape)
print(ind_FN.shape)



#%% вывод reject
reject=readData('reject.hdf5','reject')
print(reject.shape)

rej_test=reject.reshape( (reject.shape[0]*reject.shape[1]*reject.shape[2],-1) )[300000:]

print(rej_test.shape)
rej_test=rej_test.reshape(-1)
print(rej_test.shape)
print(rej_test)
vectorReject=plt.hist(rej_test, bins = [-3,-2,-1,0,1,2,3]) 
plt.title("histogram") 
plt.show()

ind_FP = np.where(( (res_test==0) & (res_pred==1)))[0] #  то есть res_test = 0 , а res_pred = 1
ind_FN = np.where(( (res_test==1) & (res_pred==0)))[0] #  то есть res_test = 1 , а res_pred = 0
print(ind_FP.shape)
print(ind_FN.shape)

errFP=np.empty(shape=(len(ind_FP)))

j=0
for i in range(len(ind_FP)):
    errFP[j]=rej_test[ind_FP[i]]
    j+=1
print(errFP)
plt.hist(errFP, bins = [-3,-2,-1,0,1,2,3]) 
plt.title("histogram") 
plt.show()
    

errFN=np.empty(shape=(len(ind_FN)))

j=0
for i in range(len(ind_FN)):
    errFN[j]=rej_test[ind_FN[i]]
    j+=1
print(errFN)
plt.hist(errFN, bins = [-3,-2,-1,0,1,2,3]) 
plt.title("histogram") 
plt.show()
#%%
print(rej_test[:10])
print(y_test[:10])


bugEl = np.where(( (rej_test<0) & (y_test==1)))[0] #  то есть res_test = 0 , а res_pred = 1
print(bugEl.shape)
print(rej_test[bugEl[0]],y_test[bugEl[0]])
#for i in range(len(rej_test)):
#    if ((rej_test[i]<0)and(y_test[i]==0)):
#        print(rej_test[i])

#%%
#print(errFP)
#plt.hist(errFP, bins = [-1,-0.9,-0.8,-0.7,0.0]) 
#plt.title("histogram") 
#plt.show()
#


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
#print(dist.shape)
#%%

vectorFP=plt.hist(distFP, bins = [0,5,10,15,20,25,30,32]) 
plt.title("histogram") 
plt.show()
#%%

vectorFN=plt.hist(distFN, bins = [0,5,10,15,20,25,30,32]) 
plt.title("histogram") 
plt.show()
#%%
ind_all=np.arange(60000)
distAll=getDistanceARR(ind_all,matrixData)
#%% 

vectorAll = plt.hist(distAll, bins = [0,5,10,15,20,25,30,32]) 
plt.title("histogram") 
plt.show()
#%%
print(vectorAll[0])
print(vectorFP[0])
relFP = vectorFP[0]/vectorAll[0]

print(relFP)
#%%
print(vectorAll[0])
print(vectorFN[0])
relFN = vectorFN[0]/vectorAll[0]

print(relFN)
#%%
print(relFN)
print(relFP)
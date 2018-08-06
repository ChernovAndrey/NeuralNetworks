#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:30:34 2018

@author: andrey
"""
#%%  
from myUtils import readData
import numpy as np
y_test=readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/mix_fixed/dataset_30_07_2018','y_test')
x_test = readData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/mix_fixed/dataset_30_07_2018','x_test')

print(x_test.shape)
print(y_test.shape)
#%%
from keras.models import load_model
model = load_model('/home/andrey/datasetsNN/landScapes/landScape_3000_32/model_AlexNet_fullConv550_01_08_2018.hdf5')
print(model.summary())
#%% только для graph cnn(3 точки)

def getPoints(matrix): #32*32
    p=np.zeros(shape=(2,2))
    k=0 # индекс по массиву p
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] ==1:
                p[k]=(i,j)
                k+=1
    return p                

p_test = getPoints(x_test)
x_test = x_test[:,:,:,:1] # 360000*32*32*4 ->360000*32*32*1
#%%
from sklearn.metrics import confusion_matrix
res_predict = model.predict([x_test,])
#%%
#%% 2 points
import matplotlib.pyplot as plt
res_predict = res_predict.reshape(-1)
def show_distribution_output():# распредление выходов видимых и невидимых точек
    ind_0 = np.where(y_test==0)
    ind_1 = np.where(y_test==1)
    plt.plot(np.sort(res_predict[ind_1]))        
show_distribution_output()
        
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
from sklearn.metrics import roc_curve, auc


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

ind_FP = np.where(( (res_test==0) & (res_pred==1)))[0] #  то есть res_test = 0 , а res_pred = 1
ind_FN = np.where(( (res_test==1) & (res_pred==0)))[0] #  то есть res_test = 1 , а res_pred = 0
print(ind_FP.shape)
print(ind_FN.shape)



#%% вывод reject
reject=readData('reject.hdf5','reject')
print(reject.shape)
count_train = 300000
rej_test=reject.reshape( (reject.shape[0]*reject.shape[1]*reject.shape[2],-1) )[count_train:]

print(rej_test.shape)
rej_test=rej_test.reshape(-1)
print(rej_test.shape)

vectorReject=plt.hist(rej_test, bins = [-3,-2,-1,0,1,2,3]) 
plt.title("histogram") 
plt.show()
#%%
ind_FP = np.where(( (res_test==0) & (res_pred==1)))[0] #  то есть res_test = 0 , а res_pred = 1
ind_FN = np.where(( (res_test==1) & (res_pred==0)))[0] #  то есть res_test = 1 , а res_pred = 0
print(ind_FP.shape)
print(ind_FN.shape)


def show_errors_samples(ind):#ind - индексы error samples
    errFP=np.empty(shape=(len(ind)))
    j=0
    for i in range(len(ind)):
        errFP[j]=rej_test[ind[i]]
        j+=1
    plt.hist(errFP, bins = [-3,-2,-1,0,1,2,3]) 
    plt.title("histogram") 
    plt.show()
    
show_errors_samples(ind_FN)
show_errors_samples(ind_FP)
#%%
def check_bug():
    bugEl = np.where(( (rej_test<0) & (y_test==1)))[0] #  то есть res_test = 0 , а res_pred = 1
    print(rej_test[bugEl[0]],y_test[bugEl[0]])


#%% для двух точек
import numpy as np
import math
print(x_test.shape)



def getDistance(p1,p2):
    return math.sqrt( (p1[0]-p2[0])*(p1[0]-p2[0])  + (p1[1]-p2[1])*(p1[1]-p2[1]) ) 

def getDistance_arrays(ind, matrixData): 
    dist = np.empty(shape=(len(ind)))
    for i in range(len(ind)):
        p = getPoints(matrixData[ind[i]][1])
        dist[i] = getDistance(p[0],p[1])
    return dist    
#        dist[i] = getDistance(p[0],p[1])

count_train = 60000
image_size = 32
count_input_martix=2 # кол-во входных матрица в сетку
              
matrixData = x_test.reshape(count_train,count_input_martix,image_size,image_size)                
print(matrixData.shape)
distFP = getDistance_arrays(ind_FP, matrixData)
distFN = getDistance_arrays(ind_FN, matrixData)
#%%
def hist_dist(dist,bins=[0,5,10,15,20,25,30,32]):    
    vector = plt.hist(dist, bins = [0,5,10,15,20,25,30,32]) 
    plt.title("histogram") 
    plt.show()
    return vector

vectorFN = hist_dist(distFN)
vectorFP = hist_dist(distFP)
    
#%%
ind_all=np.arange(count_train)
distAll=getDistance_arrays(ind_all,matrixData)
vectorAll = hist_dist(distAll)

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

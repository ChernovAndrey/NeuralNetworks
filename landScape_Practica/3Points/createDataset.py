#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 11:18:40 2018

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
path ='Landscapes_3000_32x32_clear.hdf5'
X_data = readData(path,'Landscapes')
print(X_data.shape)
#%%


def norm2(p1,p2):
    return math.sqrt(  (p2[0]-p1[0])*(p2[0]-p1[0])  +  (p2[1]-p1[1])*(p2[1]-p1[1])  )

def saveData(path,name,data):
    import h5py
    h5f = h5py.File(path, 'w')
    h5f.create_dataset(name, data=data,dtype=np.float32)
    h5f.close()

def random_points():
    p1= (random.randint(0,count_pixels-1),random.randint(0,count_pixels-1))
    p2=p1
    while(norm2(p1,p2) < math.sqrt(8)):
        p2= (random.randint(0,count_pixels-1),random.randint(0,count_pixels-1))
    return p1,p2       

from IterativeAlgorithmWithAnswer import calculateResult as calcRes_2points
import numpy as np
import random
import math

count_landscapes=3000
count_tuples=5# количество комбанаций в данном случае трех точек на одном ландшафте
count_shifts=8
count_pixels=32
count_points = 3

    

def calculateWithResult():  
    data3Points = np.zeros( ( count_landscapes, count_tuples, count_shifts,
                             count_points+1, count_pixels, count_pixels ),dtype=np.float32 )# 1 - для самого ландшафта              
    result = np.random.randint(2, size=(count_landscapes,count_tuples,count_shifts,count_points))# 1-2;2-3;3-1 
    rejects = np.zeros((count_landscapes,count_tuples,count_shifts,count_points),dtype=np.float32)# разница медлу прямой и ландшафтом
    for i in range(count_landscapes):
        if (i%1==0):
            print("START LANDSCAPE NUMBER:",i)
        for j in range(count_tuples):
            if (j%100==0):
                print("START LANDSCAPE NUMBER, count_tuples Number:",i,j)
            for k in range(count_shifts):
                image= X_data[i][j][k] 
                p1,p2, rejects[i][j][k][0]=calculatePointsWithResults(image,result[i][j][k][0])
                p3,result[i][j][k][2],result[i][j][k][1],rejects[i][j][k][2],rejects[i][j][k][1] = \
                calculatePoint3(image,p1,p2, result[i][j][k][2],result[i][j][k][1]) # так как последний это 3-1
                data3Points[i][j][k][0] = image
                data3Points[i][j][k][1][p1] = 1
                data3Points[i][j][k][2][p2] = 1
                data3Points[i][j][k][3][p3] = 1
    return data3Points, result,rejects                                           

def rand_point(p1,p2): # не равный p1 и p2
    p=(0,0)
    while True:
        p =(random.randint(0,count_pixels-1),random.randint(0,count_pixels-1))
        if ((p!=p1)and(p!=p2)):
            break
    return p 

def calculatePoint3(image,p1,p2, expResult_p1,expResult_p2): # expresult_pi=  ожидаемый результат с точкой i 
    p3 = (0,0)
    result_p1 = False
    result_p2 =False
    flagFirst=False
    count_iter=0
    while((result_p1!=expResult_p1)or(result_p2!=expResult_p2)or(flagFirst!=True)):
#        print('p1',result_p1,expResult_p1)
#        print('p2',result_p2,expResult_p2)
        flagFirst=True
#        print('p3',p3)
        p3 = rand_point(p1,p2)
        x1,y1,x2,y2,x3,y3 =p1[0],p1[1],p2[0],p2[1],p3[0],p3[1]
        z1,z2,z3=image[p1],image[p2],image[p3]
        result_p1,min_reject1=calcRes_2points( (x1,y1,z1),(x3,y3,z3),image)    
        result_p2,min_reject2=calcRes_2points( (x2,y2,z2),(x3,y3,z3),image)
        count_iter+=1
        if count_iter>=10:
#            print("sorry I can not")
            return False,p3,result_p1,result_p2,min_reject1,min_reject2
    return True,p3,result_p1,result_p2,min_reject1,min_reject2    # первый bool-это смог ли он сгеренерить по даваемому ему результату. 
def calculatePointsWithResults(matrix,expected_result):
    result = True
    flagFirst=False
    p1=(0,0)
    p2=(0,0)
    while((result!=expected_result)or(flagFirst!=True)):
        flagFirst=True
        p1,p2=random_points()
        z1=matrix[p1]
        z2=matrix[p2]
        x1,y1,x2,y2 =p1[0],p1[1],p2[0],p2[1]            
        result,min_reject=calcRes_2points( (x1,y1,z1),(x2,y2,z2),matrix)

    return p1,p2,min_reject


#%%
#dp,result,rejects = calculateWithResult()
#saveData('datasetsNN/landScapes/landScape_3000_32/3points/dataset.hdf5','input',dp)    
#saveData('datasetsNN/landScapes/landScape_3000_32/3points/dataset.hdf5','result',result)    
#saveData('datasetsNN/landScapes/landScape_3000_32/3poins/dataset.hdf5','reject',rejects)    


def calcMultiThreads(l):
    flagOne=True # если flagOne =true, то result только единица
    print(l)
    coef = 2 # кол-во созданных экземпляров в одном треде
 #   flagResExp = np.zeros(coef)
    data3Points = np.zeros( ( coef, count_tuples, count_shifts,
                             count_points+1, count_pixels, count_pixels ),dtype=np.float32 )# 1 - для самого ландшафта              
    if flagOne==False:
        result = np.random.randint(2, size=(coef,count_tuples,count_shifts,count_points))# 1-2;2-3;3-1 
    else:
        result=np.ones((coef,count_tuples,count_shifts,count_points))
    rejects = np.zeros((coef,count_tuples,count_shifts,count_points),dtype=np.float32)# разница медлу прямой и ландшафтом
    
    ind = 0    
    for i in range(coef*l, coef*(l+1)):
        if (i%1==0):
            print("START LANDSCAPE NUMBER:",i)
        for j in range(count_tuples):
            for k in range(count_shifts):
                image= X_data[i][j][k] 
                p1,p2, rejects[ind][j][k][0]=calculatePointsWithResults(image,result[ind][j][k][0])
                flagResExp,p3,result[ind][j][k][2],result[ind][j][k][1],rejects[ind][j][k][2],rejects[ind][j][k][1] = \
                calculatePoint3(image,p1,p2, result[ind][j][k][2],result[ind][j][k][1]) # так как последний это 3-1
                data3Points[ind][j][k][0] = image
                data3Points[ind][j][k][1][p1] = 1
                data3Points[ind][j][k][2][p2] = 1
                data3Points[ind][j][k][3][p3] = 1
        ind=ind+1
        
#    saveData('datasetsNN/landScapes/landScape_3000_32/3points/dataset.hdf5','input'+str(l),data3Points)    
#    saveData('datasetsNN/landScapes/landScape_3000_32/3points/dataset.hdf5','result'+str(l),result)    
#    saveData('datasetsNN/landScapes/landScape_3000_32/3poins/dataset.hdf5','reject'+str(l),rejects)    
    
    return data3Points, result,rejects  #первый флаг - надо ли записывать результат                                         

#%%
from multiprocessing import Pool
count_thr = 15
if __name__ == '__main__':
    with Pool(count_thr) as p:
      allVal = p.map(calcMultiThreads, range(count_thr)) # первый индекс кол-в потоков, второй кол-во перемен(в нашем случае3)
#%%
#allVal=np.stack(allVal,axis=0)
dp=[]
res=[]
rej=[]
for i in range(count_thr):
     dp.append(allVal[i][0])
     res.append(allVal[i][1])
     rej.append(allVal[i][2])
#%%
print(allVal[i][1].shape)     
#%%     
dp = np.stack(dp,axis=0)
res = np.stack(res,axis=0)
rej = np.stack(rej,axis=0)
#%%
dp = dp.reshape(dp.shape[0]*dp.shape[1],count_tuples,8,4,32,32)
res = res.reshape(res.shape[0]*res.shape[1],count_tuples,8,3)
rej = rej.reshape(rej.shape[0]*rej.shape[1],count_tuples,8,3)
#%%
print(dp.shape)
print(res.shape)
print(rej.shape)
#%%
dp = dp.reshape(dp.shape[0]*dp.shape[1]*dp.shape[2],4,32,32)
res = res.reshape(res.shape[0]*res.shape[1]*res.shape[2],3)
rej = rej.reshape(rej.shape[0]*rej.shape[1]*rej.shape[2],3)
#%%
dpOnes=np.zeros(dp.shape)
resOnes=np.zeros(res.shape)
rejOnes=np.zeros(rej.shape)
#%%
j=0
for i in range(len(res)):
    curRes=res[i]
    print(curRes)
    if((curRes[0]==1.0)and(curRes[1]==1.0)and(curRes[2]==1.0)):
        dpOnes[j]=dp[i]
        resOnes[j]=res[i]
        rejOnes[j]=rej[i]
        j+=1
#%%
print(j)
dpOnes=dpOnes[:j]        
resOnes=resOnes[:j]        
rejOnes=rejOnes[:j]        
#%%
#%%     
saveData('datasetOnes.hdf5','input',dp)    
saveData('resultOnes.hdf5','result',res)    
saveData('rejectOnes.hdf5','reject',rej)    
       


#%%
dp =readData('datasetOnes.hdf5','input')    
res= readData('resultOnes.hdf5','result')    
rej = readData('rejectOnes.hdf5','reject')    

print("dp",dp.shape)
print("res",res.shape)
print("rej",rej.shape)
#%%
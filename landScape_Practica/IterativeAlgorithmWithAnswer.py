#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:05:41 2018

@author: andrey
"""
#%%
import math                
def get_n(p1,p2): # единичный направляющий вектор
    v = diffVectors(p1,p2)
    norm = math.sqrt( v[0]*v[0]+v[1]*v[1] + v[2]*v[2] ) 
    return (v[0]/norm,v[1]/norm, v[2]/norm)               

def max(a,b):
    if (a>b):
        return a
    return b

def diffVectors(v1,v2):
    return (v2[0]-v1[0],v2[1]-v1[1],v2[2]-v1[2])

def multiNumber_Vector(a,v):
    return(a*v[0],a*v[1],a*v[2])

def residual(a,b):
    max_ab=  max( abs(a[0]-b[0]),abs(a[1]-b[1]))
    return max(max_ab,abs(a[2]-b[2]) )

def sumVector(v1,v2):
    return (v1[0]+v2[0],v1[1]+v2[1],v1[2]+v2[2])

def calculateResult(p1,p2, matrix):
    if ((p1[0]-p2[0]==0)and(p1[1]-p2[1]==0)and(p1[2]-p2[2]==0)):
        return True
    #flagValue = max( matrix[p1],matrix[p2] ) # если больше него то выходим
    tau=0.01 
    z = p1 # итеративная перменная.
    n = get_n(p1,p2)
#    print('n=',n)
    n = multiNumber_Vector(tau,n)
    eps=0.011
    countIter=0
    while( residual(z,p2) > eps):       
        countIter+=1
        if (countIter>100000):
            print("not work")
            return False
        z = sumVector(z,n)
#        print("z",z[0],z[1])
        _,x =math.modf(z[0])
        _,y =math.modf(z[1])
        x = int(x)
        y = int(y)
#        print("x,y",x,y)
        if ((x==p2[0])and(y==p2[1])):
#            print("countIter=",countIter)
            return True
        value = matrix[x][y]
#        print("value",value)
#        print("z[2]",z[2])
        if ( (z[2]<value) and  ((x!=p1[0])or(y!=p1[1])) ):
#            print("countIter=",countIter)
            return False
#        print(z[0],z[1],z[2])
#        print("res=",residual(z,p2))
#    print("countIter=",countIter)
    return True    
#%%
def calculatePointsWithResults(p1,p2,matrix,expected_result):
    result=calculateResult(p1,p2,matrix)    
    while(result!=expected_result):
        
        result=calculateResult(p1,p2,matrix)
        
#%%
    
import numpy as np
result = np.random.randint(2, size=(3000,15,8))


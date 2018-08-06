#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 09:19:26 2018

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
    max_ab =  max( abs(a[0]-b[0]),abs(a[1]-b[1]))
    return max(max_ab,abs(a[2]-b[2]) )

def sumVector(v1,v2):
    return (v1[0]+v2[0],v1[1]+v2[1],v1[2]+v2[2])

def calculateResult(p1,p2, matrix):
    if ((p1[0]-p2[0]==0)and(p1[1]-p2[1]==0)and(p1[2]-p2[2]==0)):
        return True
    tau=0.001 
    z = p1 # итеративная перменная.
    n = get_n(p1,p2)
    n = multiNumber_Vector(tau,n)
    eps=0.0011
    countIter=0
    while( residual(z,p2) > eps):       
        countIter+=1
        if (countIter>100000):
            print("not work")
            return False, 0.0
        z = sumVector(z,n)
        _,x =math.modf(z[0])
        _,y =math.modf(z[1])
        x = int(x)
        y = int(y)
        value = matrix[x][y]
        reject = z[2] - value
        if ((x==p2[0])and(y==p2[1])):
            return True, reject
        if ( (reject<0) and  ((x!=p1[0])or(y!=p1[1])) ):
            return False, reject
    return True,reject    



#%% testing algorithm
def test():
    import numpy 
    image1=numpy.random.randint(10,size=(3,3)) 
    x1=0
    y1=0
    p1 = (x1,y1)
    z1=image1[p1]
    x2=2
    y2=1
    p2=(x2,y2)
    z2=image1[p2]
    print("z",z1,z2)
    print(calculateResult((x1,y1,z1),(x2,y2,z2),image1))
    print(image1)
    print(image1[2][1])

#%%



    
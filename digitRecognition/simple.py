#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 12:49:55 2018

@author: andrey
"""
#%% get Data from files
from numpy import genfromtxt
#dataSample = genfromtxt('../dataset/sample_submission.csv', delimiter=',')
testDataFile = genfromtxt('dataset/test.csv', delimiter=',') #fromfile not change
trainDataFile = genfromtxt('dataset/train.csv', delimiter=',') #fromfile not change!
print("finish get data from files")

#%% cosmetic data from files
import numpy as np

testData = testDataFile
trainData = trainDataFile

print("delete titles rows:")
print(testData.shape)
print(trainData.shape)
testData =  np.delete(testData, 0, axis=0)
trainData =  np.delete(trainData, 0, axis=0)
print(testData.shape)
print(trainData.shape)

answer =trainData[:,0] #for train data; get first column

trainData =  np.delete(trainData, 0, axis=1)

print(trainData.shape)
print("answer",answer)
print(testData)
print(trainData)

print("finish cosmetic data from files")

#%% get Images
def getImages(data): # разбиение вектора с изображением, на матрицу пикселей
    dim = int(np.sqrt(data.shape[1]))
    images = np.zeros(shape=(data.shape[0],dim,dim))
    
    for i in range(data.shape[0]): # по фотографиям
        image = np.zeros(shape=(dim,dim))
        row = data[i,:] # get i rows
        for j in range(dim): 
            for k in range(dim):
                image[j][k]=row[dim*j+k]
        images[i]=image            
    return images
    #print(image.shape)

imagesTest = getImages(testData)    
imagesTrain = getImages(trainData)    
print(imagesTest.shape)
print(imagesTrain.shape)
#%% data scaling
            
scaleImTrain = imagesTrain/255.0
scaleImTest = imagesTest/255.0
#%% сигмоида
def f(x):
    return 1 / (1 + np.exp(-x))
def f_deriv(x):
    return f(x) * (1 - f(x))
#%%  mean pooling (2,2) all images
import skimage.measure
for layer0 in scaleImTrain:
    layer1=skimage.measure.block_reduce(layer0, (2,2), np.mean)
    
    
    #print(layer1.shape)
    #image 
#layer1=skimage.measure.block_reduce(layer0, (2,2), np.mean)
#print(layer1)
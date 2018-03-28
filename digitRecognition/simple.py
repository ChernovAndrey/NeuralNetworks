т#!/usr/bin/env python3
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

#%% convert answer to vector
def convertToVector(ans):
    ansVect = np.zeros(shape =(len(ans), 10))
    for i in range(len(ans)):
        ansVect[i,int(ans[i])]=1
    return ansVect

ansVect = convertToVector(answer)     
print(ansVect)
print(ansVect.shape)
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
#%% start mean pooling (2,2) all images
import skimage.measure
print(scaleImTrain.shape)
print(scaleImTest.shape)
poolTrain = np.zeros(shape=(42000,14,14))
i=0
for layer0 in scaleImTrain:
    poolTrain[i]=skimage.measure.block_reduce(layer0, (2,2), np.mean)
    i=i+1

poolTest = np.zeros(shape=(28000,14,14))
i=0
for layer0 in scaleImTest:
    poolTest[i]=skimage.measure.block_reduce(layer0, (2,2), np.mean)
    i=i+1
    
print(poolTrain.shape)   
print(poolTest.shape)   
#%% convert (m,n,n) to (m,n*n)
def convertImagesToVectors(data):
    dim = data.shape[1]
    print(dim)
    vectData = np.zeros(shape=(data.shape[0], dim*dim))
    for i in range(len(data)):
        for j in range(dim): 
            for k in range(dim):
                vectData[i][dim*j+k] =data[i][j][k]
    return vectData     
poolTrainV = convertImagesToVectors(poolTrain)
poolTestV = convertImagesToVectors(poolTest)

print(poolTestV.shape)            
print(poolTrainV.shape)            

#%% train- 90%; valid = 10%
readyTrainData = poolTrainV[:37800]
readyValData = poolTrainV[37800:]
print(readyTrainData.shape)
print(readyValData.shape)

#%% training
nn_structure = [196,44, 10]

def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = np.random.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = np.random.random_sample((nn_structure[l],))
    return W, b

def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b


def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise,
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l))
    return h, z


def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y-h_out) * f_deriv(z_out)


def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)

def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.25):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, to be used in the
            # gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func

W, b, avg_cost_func = train_nn(nn_structure, readyTrainData, ansVect)
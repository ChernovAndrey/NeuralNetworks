#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:16:57 2018

@author: andrey
"""

# %%
import h5py

h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/2/ready_data.hdf5', 'r')
data = h5f['dataset_3000'][...]
h5f.close()

h5f = h5py.File('datasetsNN/landScapes/landScape_3000_32/2/ready_res.hdf5', 'r')
result = h5f['dataset_3000'][...]
h5f.close()

print(data.shape)
print(result.shape)

count_train = 300000
x_train = data[:count_train]
x_test = data[count_train:]
y_train = result[:count_train]
y_test = result[count_train:]

# %%
import numpy as np
from neuralNetworks.landScape_Practica.IterativeAlgorithm import calculateResult


def getPoints(matrix):  # 32*32
    p = np.zeros(shape=(2, 2))
    k = 0  # индекс по массиву p
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                p[k] = (i, j)
                k += 1
    #    print(p)
    return p


def checkOneEl(input):  # input 32*32*2; answerExpec: 1
    input = input.reshape(2, 32, 32)
    matrix = input[0]
    p = getPoints(input[1])
    p1 = (p[0][0], p[0][1], matrix[p[0][0]][p[0][1]])
    p2 = (p[1][0], p[1][1], matrix[p[1][0]][p[1][1]])

    return calculateResult(p1, p2, input[0])


def checkDataset(data, ansExpec):  # проверим каждый coef элемент
    coef = 100
    n = len(data) / coef
    print(n)
    for i in range(int(n)):
        if (i % 1000 == 0):
            print("check number ", coef * i)
        j = coef * i
        ans = checkOneEl(data[j])
        if (int(ans) != int(ansExpec[j])):
            print("NOT MATCH")


# %%
checkDataset(data, result)

print(result)

# %%
import numpy as np


def mixDataset(data, result):  # not used
    for i in 10000:
        j = np.random.randint(data.shape[0])
        result[i], result[j] = result[j], result[i]
        data[i], data[j] = data[j], data[i]


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


X_train_new, y_train_new = unison_shuffled_copies(x_train, y_train)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# %%
print(X_train_new.shape)
print(y_train_new.shape)
# %%
checkDataset(X_train_new, y_train_new)


# %%
def saveData(path, name, data):
    import h5py
    h5f = h5py.File(path, 'w')
    h5f.create_dataset(name, data=data, dtype=np.float32)
    h5f.close()


saveData('datasetsNN/landScapes/landScape_3000_32/mix/trainData.hdf5', 'train_3000_', X_train_new)
saveData('datasetsNN/landScapes/landScape_3000_32/mix/trainResult.hdf5', 'result_3000', y_train_new)

# %%
print(x_test.shape)
print(y_test.shape)
saveData('datasetsNN/landScapes/landScape_3000_32/mix/testData.hdf5', 'test_3000_', x_test)
saveData('datasetsNN/landScapes/landScape_3000_32/mix/testResult.hdf5', 'resultTest_3000', y_test)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 09:35:22 2018

@author: andrey
"""

#%% генерация точек
from myUtils import readData
import numpy as np
import math
from numpy import linalg as LA
from IterativeAlgorithmWithAnswer import calculateResult
count_landscapes = 3000
count_pair_points = 15
count_shifts = 8
count_pixels = 32


path = 'datasetsNN/landScapes/landScape_3000_32/Landscapes_3000_32x32_clear.hdf5'
X_data = readData(path, 'Landscapes')
print(X_data.shape)

#%%

def random_points():
    p1 = np.random.randint(0, count_pixels - 1, 2)
    p2 = p1
    while (LA.norm(p1-p2) < math.sqrt(8)): # чтобы точки не были близки друг к другу
        p2 = np.random.randint(0, count_pixels - 1, 2)
    return p1, p2


point_data = np.zeros((count_landscapes, count_pair_points, count_shifts, count_pixels, count_pixels))
result = np.zeros((count_landscapes, count_pair_points, count_shifts))


def calculate_simple():  # простая генерация, которая не заботится о процентном соотношении образцов
    for i in range(count_landscapes):
        if (i % 200 == 0):
            print("START LANDSCAPE NUMBER:", i)
        for j in range(count_pair_points):
            for k in range(count_shifts):
                p1, p2 = random_points()
                point_data[i][j][k][p1] = 1
                point_data[i][j][k][p2] = 1
                image = X_data[i][j][k]
                z1 = image[p1]
                z2 = image[p2]
                x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
                if (calculateResult((x1, y1, z1), (x2, y2, z2), image) == True):
                    result[i][j][k] = 1


# %% with known result
import np as np


def calculateWithResult():  # генерация по ответам
    result = np.random.randint(2, size=(count_landscapes, count_pair_points, count_shifts))
    for i in range(count_landscapes):
        if (i % 10 == 0):
            print("START LANDSCAPE NUMBER:", i)
        for j in range(count_pair_points):
            for k in range(count_shifts):
                image = X_data[i][j][k]
                p1, p2 = calculatePointsWithResults(image, result[i][j][k])
                point_data[i][j][k][p1] = 1
                point_data[i][j][k][p2] = 1
    return point_data, result


def calculatePointsWithResults(matrix, expected_result):
    result = True
    flagFirst = False
    p1 = (0, 0)
    p2 = (0, 0)
    while ((result != expected_result) or (flagFirst != True)):
        flagFirst = True
        p1, p2 = random_points()
        z1 = matrix[p1]
        z2 = matrix[p2]
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        result = calculateResult((x1, y1, z1), (x2, y2, z2), matrix)

    return p1, p2


pd, res = calculateWithResult()
# %%
from myUtils import saveData

saveData('datasetsNN/landScapes/landScape_3000_32/2/points.hdf5', 'points_3000_2', pd)
saveData('datasetsNN/landScapes/landScape_3000_32/2/result.hdf5', 'results_3000_2', res)


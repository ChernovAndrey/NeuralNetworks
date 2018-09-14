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
import os
from myUtils import saveData
from multiprocessing import Pool
count_landscapes = 270
count_pair_points = 15
count_shifts = 8
count_pixels = 32
shift = 2 # cдвиг от границы при рандомной генерации точек
min_reject = 1e-03 # минимальный доступимый отклонение.
count_thread = 20
count_train = 2700 # count unequal landscapes
path = ''
save_path = 'train/'
X_data = readData(os.path.join(path,'Landscapes_05_09_2018.hdf5'), 'Landscapes')
print(X_data.shape)
#%%

def random_points():
    p1 = np.random.randint(shift, count_pixels - shift - 1, 2)
    p2 = p1
    while (LA.norm(p1-p2) < math.sqrt(8)): # чтобы точки не были близки друг к другу
        p2 = np.random.randint(shift, count_pixels - shift - 1, 2)
    return p1, p2


#point_data = np.zeros((count_landscapes, count_pair_points, count_shifts, count_pixels, count_pixels))
#result = np.zeros((count_landscapes, count_pair_points, count_shifts))
#
#
##def calculate_simple():  # простая генерация, которая не заботится о процентном соотношении образцов
##    for i in range(count_landscapes):
##        if (i % 200 == 0):
##            print("START LANDSCAPE NUMBER:", i)
##        for j in range(count_pair_points):
##            for k in range(count_shifts):
##                p1, p2 = random_points()
##                point_data[i][j][k][p1] = 1
##                point_data[i][j][k][p2] = 1
##                image = X_data[i][j][k]
##                z1 = image[p1]
##                z2 = image[p2]
##                x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
##                if (calculateResult((x1, y1, z1), (x2, y2, z2), image) == True):
##                    result[i][j][k] = 1
##

# %% with known result



def calculateWithResult():  # генерация по ответам (однопоточная)
    count_thread_land = count_landscapes//count_thread
    result = np.random.randint(2, size=(count_thread_land, count_pair_points, count_shifts))
    point_data = np.zeros((count_thread_land, count_pair_points, count_shifts, count_pixels, count_pixels))
    for i in range(count_landscapes):
        if (i % 100 == 0):
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
    reject = 99999.0
    while ((result != expected_result) or (flagFirst != True) or (abs(reject) < min_reject)):
        flagFirst = True
        p1, p2 = random_points()
        z1 = matrix[p1[0]][p1[1]]
        z2 = matrix[p2[0]][p2[1]]
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        result, reject = calculateResult((x1, y1, z1), (x2, y2, z2), matrix)
    return p1, p2, reject

def calculateWithResult_multiThread(n_thread):  # генерация по ответам (многопоточная)
    count_thread_land = count_landscapes//count_thread
    result = np.random.randint(2, size=(count_thread_land, count_pair_points, count_shifts))
    point_data = np.zeros((count_thread_land, count_pair_points, count_shifts, count_pixels, count_pixels))
    reject_data = np.zeros((count_thread_land, count_pair_points, count_shifts))
    result = np.zeros((count_thread_land, count_pair_points, count_shifts))
    
    for i in range( n_thread*count_thread_land,(n_thread+1)*count_thread_land ):
        i1 = i % count_thread_land
        print("START LANDSCAPE NUMBER:", i,' n_thread=', n_thread, 'i1=', i1)
        for j in range(count_pair_points):
            for k in range(count_shifts):
                image = X_data[i][j][k]
                p1, p2, reject_data[i1][j][k] = calculatePointsWithResults(image, result[i1][j][k])
                point_data[i1][j][k][p1] = 1
                point_data[i1][j][k][p2] = 1
                print(np.where(point_data[i1][j][k]!=0))
                print('check')
                print(point_data[i1][j][k][p1] , point_data[i1][j][k][p2] )
#    saveData(os.path.join(save_path,'points_05_09_2018_'+str(n_thread)), 'points_' + str(n_thread), point_data)
#    saveData(os.path.join(save_path,'reject_05_09_2018_'+str(n_thread)), 'reject_' + str(n_thread), reject_data)
#    saveData(os.path.join(save_path,'results_05_09_2018_'+str(n_thread)), 'results_' + str(n_thread), result)

def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]
#pd, res = calculateWithResult()
#%% merge result and prepare result to input in CNN
def divided_train_test(input_data, res, rej):
    X_train, X_test, y_train, y_test = input_data[:count_train].reshape(-1,32,32,2), input_data[count_train:].reshape(-1,32,32,2),\
                                        res[:count_train].reshape(-1), res[count_train:].reshape(-1)
    rej_train, rej_test = rej[:count_train].reshape(-1), rej[count_train:].reshape(-1)
    unison_shuffled_copies(X_train, y_train, rej_train)
    unison_shuffled_copies(X_test, y_test, rej_test)
    
    print('ready shapes')
    print('X: ', X_train.shape, '\t', X_test.shape)
    print('Y: ', y_train.shape, '\t', y_test.shape)
    print('rej: ', rej_train.shape, '\t', rej_test.shape)
    
    saveData(os.path.join(save_path,'dataset_11_09_2018'), 'X_train', X_train )
    saveData(os.path.join(save_path,'dataset_11_09_2018'), 'y_train', y_train )
    saveData(os.path.join(save_path,'dataset_11_09_2018'), 'X_test', X_test)
    saveData(os.path.join(save_path,'dataset_11_09_2018'), 'y_test', y_test)
    saveData(os.path.join(save_path,'dataset_11_09_2018'), 'rej_train', rej_train )
    saveData(os.path.join(save_path,'dataset_11_09_2018'), 'rej_test', rej_test )
    
def merging(n_files = count_thread):        
    points = np.zeros( (0, count_pair_points, count_shifts, count_pixels, count_pixels) )
    rej = np.zeros( (0, count_pair_points, count_shifts))
    res = np.zeros( (0, count_pair_points, count_shifts))
    for i in range(n_files):
        points = np.concatenate( (points,readData(os.path.join(path,'points_05_09_2018_'+str(0)), 'points_' + str(0))), axis = 0 )
        rej = np.concatenate( (rej, readData(os.path.join(path,'reject_05_09_2018_'+str(0)), 'reject_' + str(0))), axis = 0)
        res = np.concatenate( (res, readData(os.path.join(path,'results_05_09_2018_'+str(0)), 'results_' + str(0))), axis = 0)
    points = np.expand_dims(points, axis = -1)
    landscapes = X_data.copy()  
    landscapes = np.expand_dims(landscapes, axis=-1)
    input_data = np.concatenate( (landscapes, points), axis = -1)
    divided_train_test(input_data, res, rej) 
    
#    saveData(os.path.join(path,'input_11_09_2018'), 'input', input_data)
#    saveData(os.path.join(path,'reject_11_09_2018'), 'reject' , rej)
#    saveData(os.path.join(path,'results_11_09_2018'), 'results', res)


#%%

if __name__ == '__main__':
#    with Pool(count_thread) as p:
#        p.map(calculateWithResult_multiThread, range(count_thread))  # первый индекс кол-в потоков, второй кол-во перемен(в нашем случае3) 
    merging()


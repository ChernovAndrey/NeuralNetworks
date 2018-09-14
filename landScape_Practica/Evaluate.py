#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:30:34 2018

@author: andrey
"""
# %%
from myUtils import readData
import numpy as np
from sklearn.metrics import confusion_matrix
import math
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc

path_to_dataset = '/home/andrey/datasetsNN/landScapes/landScape_sentember/dataset_11_09_2018_test.hdf5'
path_to_model = '/home/andrey/datasetsNN/landScapes/landScape_3000_32/model_AlexNet_fullConv550_01_08_2018.hdf5'
name_y_test = 'y_test'
name_x_test = 'X_test'
name_rej_test = 'rej_test'
count_points = 2  # реализовано только для двух и трех точек на ландшафте.
count_input_martix = 2  # кол-во входных матрица в сетку
count_train = 300000
count_train = 60000
image_size = 32


# %% только для graph cnn(3 точки)

def getPoints(matrix):  # 32*32
    p = np.zeros(shape=(2, 2))
    k = 0  # индекс по массиву p
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                p[k] = (i, j)
                k += 1
    return p


# %% 2 points
def show_distribution_output(res_predict, y_test):  # распредление выходов видимых и невидимых точек
    if count_points != 2:
        print('эта функция только для двух точек')
        return

    ind_0 = np.where(y_test == 0)
    ind_1 = np.where(y_test == 1)
    plt.hist(res_predict[ind_0], bins=list(np.arange(0.0, 1.0, 0.05)))
    plt.hist(res_predict[ind_1], bins=list(np.arange(0.0, 1.0, 0.05)))
    plt.title("histogram")
    plt.show()


#    print(len(ind_0[0]))
#    print(len(ind_1[0]))
#    plt.plot(np.sort(res_predict[ind_1]))        
# %%
def getInt(vector):  # zero or one
    res = np.zeros(vector.shape, dtype=int)
    for i in range(len(vector)):
        if (vector[i] >= 0.5):
            res[i] = 1
        else:
            res[i] = 0
    return res


def ROC_plot(y_test, res_predict):
    fpr, tpr, thr = roc_curve(y_test, res_predict)
    roc_auc = auc(fpr, tpr)

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


# %% вывод reject
def get_and_show_reject():
    reject = readData('reject.hdf5', 'reject')
    print(reject.shape)
    rej_test = reject.reshape((reject.shape[0] * reject.shape[1] * reject.shape[2], -1))[count_train:]
    rej_test = rej_test.reshape(-1)
    print(rej_test.shape)
    plt.hist(rej_test, bins=[-3, -2, -1, 0, 1, 2, 3])
    plt.title("histogram")
    plt.show()
    return rej_test


# %%
def get_errors_index(y_test, res_predict):
    ind_FP = np.where(((y_test == 0) & (res_predict == 1)))[0]  # то есть y_test = 0 , а res_predict = 1
    ind_FN = np.where(((y_test == 1) & (res_predict == 0)))[0]  # то есть y_test = 1 , а res_predict = 0
    print('ind_FP shape: ', ind_FP.shape)
    print('ind_FN shape: ', ind_FN.shape)
    return ind_FP, ind_FN


def show_errors_samples(ind, rej_test):  # ind - индексы error samples
    errFP = np.empty(shape=(len(ind)))
    j = 0
    for i in range(len(ind)):
        errFP[j] = rej_test[ind[i]]
        j += 1
    plt.hist(errFP, bins=[-3, -2, -1, 0, 1, 2, 3])
    plt.title("histogram")
    plt.show()


# %%
def check_bug(rej_test, y_test):  # проверка датасета на ошибки
    bugEl1 = np.where(((rej_test < 0) & (y_test == 1)))[0]  
    print(bugEl1)
    print(len(bugEl1))
#    print(rej_test[bugEl1[0]], y_test[bugEl1[0]])
#
    bugEl2 = np.where(((rej_test > 0) & (y_test == 0)))[0]  
    print(len(bugEl2))
#    print(rej_test[bugEl2[0]], y_test[bugEl2[0]])


# %% для двух точек


def getDistance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def getDistance_arrays(ind, matrixData):
    dist = np.empty(shape=(len(ind)))
    print(matrixData.shape)
    for i in range(len(ind)):
        p = getPoints(matrixData[ind[i]][1])
        dist[i] = getDistance(p[0], p[1])
    return dist


#        dist[i] = getDistance(p[0],p[1])


# %%
def hist_dist(dist, bins=[0, 5, 10, 15, 20, 25, 30, 32]):
    vector = plt.hist(dist, bins=[0, 5, 10, 15, 20, 25, 30, 32])
    plt.title("histogram")
    plt.show()
    return vector


def show_hist_dist(matrixData):  # сравнение распределения расстояния между точками для всех результатов и для ошибочных
    ind_all = np.arange(count_train)
    distAll = getDistance_arrays(ind_all, matrixData)
    vectorAll = hist_dist(distAll)
    distFP = getDistance_arrays(ind_FP, matrixData)
    distFN = getDistance_arrays(ind_FN, matrixData)
    vectorFN = hist_dist(distFN)
    vectorFP = hist_dist(distFP)
    relFP = vectorFP[0] / vectorAll[0]
    print('relFP= ', relFP)
    relFN = vectorFN[0] / vectorAll[0]
    print('relFN= ', relFN)


# %%

if __name__ == "__main__":
    x_test = readData(path_to_dataset, name_x_test)
    y_test = readData(path_to_dataset, name_y_test)
    rej_test = readData(path_to_dataset, name_rej_test)
    print('x_test shape: ', x_test.shape)
    print('y_test shape: ', y_test.shape)
    print('rej_test shape: ', rej_test.shape)
    

    model = load_model(path_to_model)
    print(model.summary())

    #    для graphCNN
    #        p_test = getPoints(x_test)
    #        x_test = x_test[:,:,:,:1] # 360000*32*32*4 ->360000*32*32*1 (убираем матрицы точек)

    res_predict = model.predict([x_test, ])
    res_predict = res_predict.reshape(-1)
    y_test = np.reshape(y_test, (-1))
    show_distribution_output(res_predict, y_test)
    ROC_plot(y_test, res_predict)
    res_predict = getInt(res_predict)
    y_test = getInt(y_test)

    confusion_matrix(y_test, res_predict)
    print("val accuracy", accuracy_score(y_test, res_predict))
    print("Точность:", precision_score(y_test, res_predict))
    print("Полнота:", recall_score(y_test, res_predict))
    print("F1:", f1_score(y_test, res_predict))

    ind_FP, ind_FN = get_errors_index(y_test, res_predict)
    #    show_errors_samples(ind_FP) by reject
    #    show_errors_samples(ind_FN)

    matrixData = x_test.reshape(count_train, count_input_martix, image_size, image_size)
    print('matrixData shape = ', matrixData.shape)
    show_hist_dist(matrixData)
#%% kek

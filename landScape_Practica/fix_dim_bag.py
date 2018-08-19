#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:25:16 2018

@author: andrey
"""

# %%
from myUtils import *

x_train, y_train, x_test, y_test = getTestData2Points()
# %%
import numpy as np


def fixed(xdata):
    for i in range(len(xdata)):
        if i % 1000 == 0:
            print(i)
        temp = np.reshape(xdata[i], (2, 32, 32))
        xdata[i] = np.moveaxis(temp, [0, 1, 2], [2, 0, 1])
    return xdata


# fix_x_test = fixed(x_test)
fix_x_train = fixed(x_train)
# %%
# saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/mix_fixed/dataset_30_07_2018','x_test',fix_x_test)
saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/mix_fixed/dataset_30_07_2018', 'x_train', fix_x_train)
saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/mix_fixed/dataset_30_07_2018', 'y_train', y_train)
saveData('/home/andrey/datasetsNN/landScapes/landScape_3000_32/mix_fixed/dataset_30_07_2018', 'y_test', y_test)

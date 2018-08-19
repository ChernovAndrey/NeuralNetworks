#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:05:41 2018

@author: andrey
"""
# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 09:19:26 2018

@author: andrey
"""

# %%
import math
import numpy as np
from numpy import linalg as LA


def get_n(p1, p2):  # единичный направляющий вектор
    v = p2 - p1
    return v/LA.norm(v)


def residual(a, b):
    return np.max(np.abs(a - b))


def calculateResult(p1, p2, matrix):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    if (np.all( p1-p2 == 0 )):
        return True
    tau = 0.0001
    z = p1  # итеративная перменная.
    n = get_n(p1, p2)
    n = tau*n
    eps = 0.00011
    countIter = 0
    min_reject = 9999.0 # заведомо большое значение
    while (residual(z, p2) > eps):
        countIter += 1
        if (countIter > 1000000):
            print("not work")
            return False, 0.0
        z = z+n
        _, x = math.modf(z[0])
        _, y = math.modf(z[1])
        x = int(x)
        y = int(y)
        value = matrix[x][y]
        reject = z[2] - value
        if abs(min_reject) > abs(reject):
            min_reject = reject
        if ((x == p2[0]) and (y == p2[1])):
            return True, min_reject
        if ((reject < 0) and ((x != p1[0]) or (y != p1[1]))):
            return False, min_reject
    return True, min_reject


def calculatePointsWithResults(p1, p2, matrix, expected_result):
    result = calculateResult(p1, p2, matrix)
    while (result != expected_result):
        result = calculateResult(p1, p2, matrix)


# %% testing algorithm
def test():
    import numpy
    image1 = numpy.random.randint(10, size=(3, 3))
    p1 = np.zeros(2,int)    
    z1 = image1[p1]
    p2 = np.array([1,2])
    z2 = image1[p2]
    
    print("z", z1, z2)
    print(calculateResult(np.append(p1,z1),np.append(p2,z2), image1))
    print(calculateResult(np.append(p2,z2),np.append(p1,z1), image1))
    print(image1)
    print(image1[2][1])
#test()

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:36:25 2018

@author: Austin Hsu
"""

import pandas as pd
import numpy as np
import sys

w = np.load("generative_w.npy")
b = np.load("generative_b.npy")

def sigmoid(x):
    return 1/(1 + np.exp(-(np.dot(w,x.reshape((23,1)))+b))[0,0])

x_test = pd.read_csv(sys.argv[3])
x_test = x_test.values
y_test_list = []

for i in x_test:
    y_test_list.append(0 if sigmoid(i) >= 0.5 else 1)

with open(sys.argv[4], 'w') as f:
    print('id,value', file=f)
    for i in range(len(y_test_list)):
        print('id_%d,%d' % (i, y_test_list[i]), file=f) 
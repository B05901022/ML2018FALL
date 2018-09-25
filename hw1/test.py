# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 22:44:40 2018

@author: Austin Hsu
"""

import pandas as pd
import numpy as np

data_test = pd.read_csv('test.csv', engine = 'python', header = None)
data_test = data_test.values

para_w = np.load("trained_w.npy")
para_b = np.load("trained_b.npy")
para_mean = np.load("data_mean.npy")
para_std = np.load("data_std.npy")

data_list = [[] for i in range(18)] 
   
for i in range(data_test.shape[0]):
    temp_list = data_test[i][2:].tolist()
    if i%18 == 10:
        for j in range(len(temp_list)):
            if temp_list[j] == 'NR':temp_list[j] = '0'
    data_list[i%18] += [(eval(k)-para_mean[i%18])/para_std[i%18] for k in temp_list]

data_array = np.array(data_list)
data_arrays_list = [data_array[:,i:i+9] for i in range(0,data_array.shape[1],9)]

prediction_list = []
for i in data_arrays_list:
    prediction_list.append((para_b + np.sum(para_w * i)).tolist()[0][0])

with open('result.csv', 'w') as f:
    print('id,value', file=f)
    for i in range(len(prediction_list)):
        print('id_%d,%f' % (i, prediction_list[i]), file=f) 
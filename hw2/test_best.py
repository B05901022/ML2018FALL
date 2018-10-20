# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 15:22:16 2018

@author: Austin Hsu
"""

import pandas as pd
import numpy as np

#np.set_printoptions(suppress=True)

para_w = np.load("trained_w.npy")
para_bias = np.load("trained_b.npy")
d_mean = np.load("data_mean.npy")
d_std = np.load("data_std.npy")

x_test = pd.read_csv('test_x.csv', engine = 'python')
x_test = x_test.values

#one hot
one_hot_columns = [1,2,3,5,6,7,8,9,10]

def one_hot(input_array):
    one_hotted_list = []
    for i in input_array:
        temp_list = []
        for j in range(i.shape[0]):
            if j in one_hot_columns:
                if   j == 1:#SEX
                    if   i[j] == 1:temp_list += [0,1]
                    elif i[j] == 2:temp_list += [1,0]
                elif j == 2:#EDUCATION
                    if   i[j] == 0:temp_list += [0,0,0,0,0,0,1]
                    elif i[j] == 1:temp_list += [0,0,0,0,0,1,0]
                    elif i[j] == 2:temp_list += [0,0,0,0,1,0,0]
                    elif i[j] == 3:temp_list += [0,0,0,1,0,0,0]
                    elif i[j] == 4:temp_list += [0,0,1,0,0,0,0]
                    elif i[j] == 5:temp_list += [0,1,0,0,0,0,0]
                    elif i[j] == 6:temp_list += [1,0,0,0,0,0,0]
                elif j == 3:#MARRIAGE
                    if   i[j] == 0:temp_list += [0,0,0,1]
                    elif i[j] == 1:temp_list += [0,0,1,0]
                    elif i[j] == 2:temp_list += [0,1,0,0]
                    elif i[j] == 3:temp_list += [1,0,0,0]
                else:#PAY_0,PAY_2~PAY_6
                    if   i[j] ==-2:temp_list += [0,0,0,0,0,0,0,0,0,0,1]
                    elif i[j] ==-1:temp_list += [0,0,0,0,0,0,0,0,0,1,0]
                    elif i[j] == 0:temp_list += [0,0,0,0,0,0,0,0,1,0,0]
                    elif i[j] == 1:temp_list += [0,0,0,0,0,0,0,1,0,0,0]
                    elif i[j] == 2:temp_list += [0,0,0,0,0,0,1,0,0,0,0]
                    elif i[j] == 3:temp_list += [0,0,0,0,0,1,0,0,0,0,0]
                    elif i[j] == 4:temp_list += [0,0,0,0,1,0,0,0,0,0,0]
                    elif i[j] == 5:temp_list += [0,0,0,1,0,0,0,0,0,0,0]
                    elif i[j] == 6:temp_list += [0,0,1,0,0,0,0,0,0,0,0]
                    elif i[j] == 7:temp_list += [0,1,0,0,0,0,0,0,0,0,0]
                    elif i[j] == 8:temp_list += [1,0,0,0,0,0,0,0,0,0,0]
            else:
                temp_list.append(i[j])
        one_hotted_list.append(np.array(temp_list))
    return np.array(one_hotted_list)

x_test = one_hot(x_test)
x_test = x_test.astype('float64')

def feature_scaling(input_array):
    scaled_array = input_array
    for i in range(scaled_array.shape[1]):
        if i not in list(range(1,14))+list(range(15,81)):
            scaled_array[:,i] = (scaled_array[:,i] - d_mean[i]) / d_std[i]
        else:
            scaled_array[:,i] = scaled_array[:,i] #- feature_mean[i]
    return scaled_array

x_test = feature_scaling(x_test)

def sigmoid(w_array, bias, input_array):
    z = np.sum(np.multiply(w_array, input_array)) + bias
    sigmoid_result = 1/(1+np.exp(-z))
    return sigmoid_result

y_test = [0 if sigmoid(para_w, para_bias, i) >= 0.5 else 1 for i in x_test]


with open('result_best.csv', 'w') as f:
    print('id,Value', file = f)
    for i in range(len(y_test)):
        print('id_%d,%d' % (i,y_test[i]), file = f)

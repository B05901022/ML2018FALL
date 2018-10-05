# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 22:35:44 2018

@author: Austin Hsu
"""

#read file
import pandas as pd
import numpy as np

data_train = pd.read_csv('train.csv', engine = 'python')
data_train = data_train.values

#combine data value
data_list = [[] for i in range(18)] 
   
for i in range(data_train.shape[0]):
    temp_list = data_train[i][3:].tolist()
    if i%18 == 10:
        for j in range(len(temp_list)):
            if temp_list[j] == 'NR':temp_list[j] = '0'
    data_list[i%18] += [eval(k) for k in temp_list]

data_array = np.array(data_list)
y_data_array = data_array[9][:]

#Garbage data filter
def is_garbage(input_array):
    garbage_list = []
    for i in range(input_array.shape[1]):
        if (input_array[:,i] == np.zeros([18,1])).all():
            garbage_list.append(i)
        
    return tuple(garbage_list)

garbage_tuple = is_garbage(data_array)
data_array = np.delete(data_array, garbage_tuple, axis = 1)

#Preprocessing data
p_data_list = []
data_mean = np.mean(data_array, axis = 1)
data_std = np.std(data_array, axis = 1)
for i in range(data_array.shape[0]):
    p_data_list.append((data_array[i]-data_mean[i])/data_std[i])

p_data_array = np.array(p_data_list)

for i in garbage_tuple:
    p_data_array = np.insert(p_data_array, i, 0, axis = 1)

#stochastic gradient descent preprocess
"""
Reference:Beyond SGD: Gradient Descent with Momentum and Adaptive Learning Rate
https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
"""
def p_sgd(input_array, input_y_array):
    x_train = []
    y_train = []
    for i in range(9,input_array.shape[1]):
        x_train.append(input_array[:,i-9:i])
        y_train.append(input_y_array[i])
    xy_train = []
    for i in range(len(x_train)):
        x_mini = x_train[i]
        y_mini = y_train[i]
        xy_train.append((np.array(x_mini),np.array(y_mini)))
    return np.array(xy_train)

#20 days per cut & stochastic gradient descent preprocess(batch_size = 1)
t_data_list = []
for i in range(12):
    temp_mb = p_sgd(p_data_array[:,i*480:(i+1)*480], y_data_array[i*480:(i+1)*480]).tolist()
    for j in temp_mb: 
        t_data_list.append(j)
        
t_data_array = np.array(t_data_list)

#delete garbage inputs from garbage_tuple
garbage_input_list = []
for i in garbage_tuple:
     temp = i%480
     low_bound = temp-9
     high_bound = temp+8
     if low_bound < 0:low_bound = 0
     if high_bound > 480:high_bound = 480
     temp_list = list(range(low_bound, high_bound+1-8))
     garbage_input_list +=  [j+480*(i//480) for j in temp_list]
garbage_input_tuple = tuple(set(garbage_input_list))

t_data_array = np.delete(t_data_array, garbage_input_tuple, axis = 0)     

#loss function
def loss(w_array, bias, input_array):
    """
    w_array:18*9 matrix
    bias:1*1 matrix
    input_array:n*(18*9) matrix, 1*1 matrix 
    """
    total_loss = 0
    for i in input_array:
        temp = 0
        temp += np.sum(np.multiply(w_array, i[0]))
        temp += bias[0]
        total_loss += (i[1] - temp) ** 2
    return (total_loss/input_array.shape[0])**(0.5)

#dL/dw and dL/db
def d_loss(w_array, bias, input_array):
    """
    inputs:
        w_array:18*9 matrix
        bias:1*1 matrix
        input_array:n*(18*9) matrix, 1*1 marix
    outputs:
        total_w:18*9 matrix
        total_b:1*1 matrix
    """
    total_w = np.zeros([18,9])
    total_b = np.zeros([1,1])
    temp = 0
    temp += np.sum(np.multiply(w_array, input_array[0]))
    temp += bias[0]
    temp_loss = input_array[1]-temp
    total_w += -2*temp_loss*input_array[0]
    total_b += -2*temp_loss
    return total_w, total_b
    
#Adam implementation
"""
Reference:ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION
https://arxiv.org/pdf/1412.6980.pdf
"""
###Adam parameters
para_w = np.random.rand(18,9)
para_bias = np.random.rand(1,1)
para_alpha = 0.002#0.0015 too large
para_beta_1 = 0.6#0.8
para_beta_2 = 0.9#0.999
para_epsilon = 1e-7#1e-8

###epoch needed
para_epoch = 500

#best record
def parameter_keep(history, new_v):
    """
    input:para_history
    output:new validation loss
    """
    
    if history[2] > new_v:
        return True
        
    else:
        return False

###Adam
def Adam(alpha, beta_1, beta_2, epsilon, w_array, bias, input_array, epoch, valid):
    ###Adam initialization
    para_m_w = np.zeros([18,9])
    para_m_b = np.zeros([1,1])
    para_v_w = np.zeros([18,9])
    para_v_b = np.zeros([1,1])
    para_t = 0
    para_w_array = w_array
    para_b_array = bias
    para_history = [para_w_array,para_b_array,10000]#best record
    for i in range(epoch):
        
        for j in input_array:
            para_t += 1
            g_t_w, g_t_b = d_loss(para_w_array, para_b_array, j)
            para_m_w = beta_1 * para_m_w + (1-beta_1) * g_t_w
            para_m_b = beta_1 * para_m_b + (1-beta_1) * g_t_b
            para_v_w = beta_2 * para_v_w + (1-beta_2) * g_t_w**2
            para_v_b = beta_2 * para_v_b + (1-beta_2) * g_t_b**2
            para_m_w_hat = para_m_w / (1-beta_1**para_t)
            para_m_b_hat = para_m_b / (1-beta_1**para_t)
            para_v_w_hat = para_v_w / (1-beta_2**para_t)
            para_v_b_hat = para_v_b / (1-beta_2**para_t)
            para_w_array = para_w_array - alpha * para_m_w_hat / (para_v_w_hat ** 0.5 + epsilon)
            para_b_array = para_b_array - alpha * para_m_b_hat / (para_v_b_hat ** 0.5 + epsilon)
        
        training_loss = loss(para_w_array, para_b_array, input_array)
        validation_loss = loss(para_w_array, para_b_array, valid)
        print("epoch:%d training loss = %f validation loss = %f" % (i+1,training_loss,validation_loss))
        if parameter_keep(para_history, validation_loss):
            para_history = [para_w_array, para_b_array, validation_loss]    
    return para_history[0], para_history[1]#para_w_array, para_b_array

#validation
"""
def validation(input_array):
    array_length = input_array.shape[0]
    random_num = np.arange(input_array.shape[0])
    np.random.shuffle(random_num)
    input_array = [input_array[i] for i in random_num]
    return np.array(input_array[:int(0.2*array_length)]), np.array(input_array[int(0.2*array_length):])
"""
def validation(input_array):
    array_length = input_array.shape[0]
    validation_set = input_array[:int(0.1*array_length)]
    training_set = input_array[int(0.1*array_length):]
    random_num = np.arange(training_set.shape[0])
    np.random.shuffle(random_num)
    training_set = [training_set[i] for i in random_num]
    return validation_set, np.array(training_set)


#noise adding
def noise_add(input_array, mu, sigma, amount):
    output_list = [input_array[i] for i in range(input_array.shape[0])]
    for j in range(amount):
        for i in range(input_array.shape[0]):
            noise = np.random.normal(mu, sigma, [18,9])
            output_list.append(np.array([input_array[i][0]+noise, input_array[i][1]]))
    return np.array(output_list)
t_data_array = noise_add(t_data_array, 0, 0.1, 1)

#early "GIVEUP"
initial_loss = 10000.00

def Adam_check(alpha, beta_1, beta_2, epsilon, w_array, bias, input_array, valid):
    para_m_w = np.zeros([18,9])
    para_m_b = np.zeros([1,1])
    para_v_w = np.zeros([18,9])
    para_v_b = np.zeros([1,1])
    para_t = 1#not 0
    para_w_array = w_array
    para_b_array = bias
    for i in input_array:
        g_t_w, g_t_b = d_loss(para_w_array, para_b_array, j)
        para_m_w = beta_1 * para_m_w + (1-beta_1) * g_t_w
        para_m_b = beta_1 * para_m_b + (1-beta_1) * g_t_b
        para_v_w = beta_2 * para_v_w + (1-beta_2) * g_t_w**2
        para_v_b = beta_2 * para_v_b + (1-beta_2) * g_t_b**2
        para_m_w_hat = para_m_w / (1-beta_1**para_t)
        para_m_b_hat = para_m_b / (1-beta_1**para_t)
        para_v_w_hat = para_v_w / (1-beta_2**para_t)
        para_v_b_hat = para_v_b / (1-beta_2**para_t)
        para_w_array = para_w_array - alpha * para_m_w_hat / (para_v_w_hat ** 0.5 + epsilon)
        para_b_array = para_b_array - alpha * para_m_b_hat / (para_v_b_hat ** 0.5 + epsilon)
    return loss(para_w_array, para_b_array, valid)
    
while initial_loss > 50.0:
    validation_array, training_array = validation(t_data_array)
    initial_loss = Adam_check(para_alpha, para_beta_1, para_beta_2, para_epsilon, para_w, para_bias, training_array, validation_array)

#executing Adam
para_w, para_bias = Adam(para_alpha, para_beta_1, para_beta_2, para_epsilon, para_w, para_bias, training_array, para_epoch, validation_array)

#Save model & update_parameter(under condition that trained_w.py and trained_b.py exist)
def update_parameter(result_parameters, load_parameters):   
    if loss(result_parameters[0], result_parameters[1], validation_array) < loss(load_parameters[0], load_parameters[1], validation_array):
        return True
    else:
        return False

"""

np.save("trained_w.npy", para_w)
np.save("trained_b.npy", para_bias)
np.save("data_mean.npy", data_mean)
np.save("data_std.npy", data_std)

"""  
para_w_load = np.load("trained_w.npy")
para_b_load = np.load("trained_b.npy")

if update_parameter([para_w, para_bias], [para_w_load, para_b_load]):
    np.save("trained_w.npy", para_w)
    np.save("trained_b.npy", para_bias)
    np.save("data_mean.npy", data_mean)
    np.save("data_std.npy", data_std)


#checking
with open("record.csv", "a") as f:
    print("%f,%f"%(loss(para_w, para_bias, training_array),loss(para_w, para_bias, validation_array)), file = f)
    

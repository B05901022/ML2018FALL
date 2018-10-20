# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:40:24 2018

@author: Austin Hsu
"""

#######################
##                   ##
##Logistic Regression##
##                   ##
#######################

import pandas as pd
import numpy as np

#np.set_printoptions(suppress=True)

x_train = pd.read_csv('train_x.csv')
x_train = x_train.values

y_train = pd.read_csv('train_y.csv')
y_train = y_train.values

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

x_train = one_hot(x_train)
x_train = x_train.astype('float64')

#feature scaling:
def feature_scaling(input_array):
    feature_mean = np.array([np.mean(input_array[:,i]) for i in range(input_array.shape[1])])
    feature_std  = np.array([np.std( input_array[:,i]) for i in range(input_array.shape[1])])
    #unscaled_list = [i for i in range(feature_std.shape[0]) if feature_std[i] == 0]
    scaled_array = input_array
    for i in range(scaled_array.shape[1]):
        if i not in list(range(1,14))+list(range(15,81)):#set(unscaled_list):
            scaled_array[:,i] = (scaled_array[:,i] - feature_mean[i]) / feature_std[i]
        else:
            #if i not in set(list(range(1,14))+list(range(15,81))):
            scaled_array[:,i] = scaled_array[:,i] #- feature_mean[i]
    return scaled_array, feature_mean, feature_std

f_x_train = feature_scaling(x_train)

d_mean  = f_x_train[1]
d_std   = f_x_train[2]
x_train = f_x_train[0]
    
d_train = np.array([[np.array([j for j in x_train[i]]).reshape((93,1)), y_train[i][0]] for i in range(x_train.shape[0])])

def sigmoid_limiter(sigmoid_value):
    if sigmoid_value >= 0.99999999:
        return 0.99999999
    elif sigmoid_value <= 0.00000001:
        return 0.00000001
    else:
        return sigmoid_value
        
#sigmoid
def sigmoid(w_array, bias, input_array):
    z = np.sum(np.multiply(w_array, input_array)) + bias
    sigmoid_result = 1/(1+np.exp(-z))
    return sigmoid_result##sigmoid_limiter(sigmoid_result)#

#cross entropy
def cross_entropy(w_array, bias, input_array, label, activation_function = 'sigmoid'):
    if activation_function == 'sigmoid':
        return -1*(label * np.log(sigmoid(w_array, bias, input_array)) + (1-label) * np.log(1-sigmoid(w_array, bias, input_array)))    

               
def loss(w_array, bias, input_array, label_array, activation_function = 'sigmoid'):
    #input_array is batched
    total_loss = 0
    batch_size = input_array.shape[0]
    for i in range(batch_size):
        total_loss += cross_entropy(w_array, bias, input_array[i], label_array[i], activation_function)
    return total_loss/batch_size

def accuracy(w_array, bias, input_array, label_array):
    result_train = [1 if sigmoid(w_array, bias, i) >= 0.5 else 0 for i in input_array]
    acc_train = [1 if result_train[i] == label_array[i] else 0 for i in range(len(result_train))]
    return sum(acc_train)/len(acc_train)

def d_loss(w_array, bias, input_array, label_array, activation_function = 'sigmoid'):
    #input_array is batched
    total_w = 0
    total_b = 0
    batch_size = input_array.shape[0]
    for i in range(batch_size):
        total_w -= (label_array[i]-sigmoid(w_array, bias, input_array[i]))*input_array[i]
        total_b -= (label_array[i]-sigmoid(w_array, bias, input_array[i]))
    return total_w, total_b

def array_batch(input_array, batch_size):
    return np.array([input_array[batch_size*i:batch_size*i+batch_size] for i in range(int(input_array.shape[0]/batch_size))])

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


def Adam(alpha, beta_1, beta_2, epsilon, w_array, bias, input_array, epoch, valid):
    ###Adam initialization
    
    para_m_w = np.zeros([93,1])
    para_m_b = 0
    para_v_w = np.zeros([93,1])
    para_v_b = 0
    para_t = 0
    
    para_w_array = w_array
    para_b_array = bias
    para_history = [para_w_array,para_b_array,10000]#best record
    batched_input_array = array_batch(input_array, 50)
    for i in range(epoch):
        """
        para_m_w = np.zeros([93,1])
        para_m_b = 0
        para_v_w = np.zeros([93,1])
        para_v_b = 0
        para_t = 0
        """
        for j in batched_input_array:
            #batched_input_array.shape == (400,50,2)
            para_t += 1
            g_t_w, g_t_b = d_loss(para_w_array, para_b_array, j[:,0], j[:,1])
            para_m_w = beta_1 * para_m_w + (1-beta_1) * g_t_w
            para_m_b = beta_1 * para_m_b + (1-beta_1) * g_t_b
            para_v_w = beta_2 * para_v_w + (1-beta_2) * g_t_w**2
            para_v_b = beta_2 * para_v_b + (1-beta_2) * g_t_b**2
            para_m_w_hat = para_m_w / (1-beta_1**para_t)
            para_m_b_hat = para_m_b / (1-beta_1**para_t)
            para_v_w_hat = para_v_w / (1-beta_2**para_t)
            para_v_b_hat = para_v_b / (1-beta_2**para_t)
            #print(para_w_array[0,0])
            #print(g_t_w[0,0])
            #print((- alpha * para_m_w_hat / (para_v_w_hat ** 0.5 + epsilon))[0,0])
            #print()
            para_w_array = para_w_array - alpha * para_m_w_hat / (para_v_w_hat ** 0.5 + epsilon)
            para_b_array = para_b_array - alpha * para_m_b_hat / (para_v_b_hat ** 0.5 + epsilon)
        training_loss   = loss(para_w_array, para_b_array, input_array[:,0], input_array[:,1])
        validation_loss = loss(para_w_array, para_b_array, valid[:,0], valid[:,1])
        training_acc    = accuracy(para_w_array, para_b_array, input_array[:,0], input_array[:,1])
        validation_acc  = accuracy(para_w_array, para_b_array, valid[:,0], valid[:,1])
        print("epoch:%d training loss = %f validation loss = %f training acc = %f validation acc = %f" % (i+1,training_loss,validation_loss, training_acc, validation_acc))
        if parameter_keep(para_history, validation_loss):
            para_history = [para_w_array, para_b_array, validation_loss]
    return para_history[0], para_history[1]#para_w_array, para_b_array#

def cross_validation(input_array_length, N):
    data_length = input_array_length//N
    random_num = np.arange(input_array_length)
    np.random.shuffle(random_num)
    validation_index = [(random_num[i*data_length:i*data_length+data_length],np.append(random_num[:i*data_length], random_num[i*data_length+data_length:])) for i in range(N)]
    return validation_index
    
def cross_validation_slice(input_array, slice_tuple):
    validation = [input_array[i] for i in slice_tuple[0]]
    training = [input_array[i] for i in slice_tuple[1]]
    return np.array(validation), np.array(training)

#Parameters
para_w_i = np.random.randn(93,1)
para_b_i = 1

para_alpha = 0.001#0.001
para_beta_1 = 0.9#0.9
para_beta_2 = 0.999#0.999
para_epsilon = 1e-8#1e-8
para_epoch = 50

cross_validation_tuples = cross_validation(d_train.shape[0], 10)
para_wb_list = []
record_list = []
cnt = 1
for m in cross_validation_tuples:
    print('segment:%d'%cnt)
    cnt += 1
    validation_array, training_array = cross_validation_slice(d_train, m)
    para_w, para_bias =para_w_i, para_b_i
    para_w, para_bias = Adam(para_alpha, para_beta_1, para_beta_2, para_epsilon, para_w, para_bias, training_array, para_epoch, validation_array)
    current_loss = loss(para_w, para_bias, validation_array[:,0], validation_array[:,1]), loss(para_w, para_bias, training_array[:,0], training_array[:,1])
    para_wb_list.append((para_w, para_bias, current_loss[0]))
    record_list.append([current_loss[0], current_loss[1]])
    
selected_wb_list = sorted(para_wb_list, key = lambda x: x[2])[:3]
para_w = (selected_wb_list[0][0] * 2 + selected_wb_list[1][0] * 2 + selected_wb_list[2][0] * 1)/5.0
para_bias = (selected_wb_list[0][1] * 2 + selected_wb_list[1][1] * 2 + selected_wb_list[2][1] * 1)/5.0

print(' validation loss  training loss')
for i in record_list:
    for j in i:
        print(j, end=" ")
    print()
print(loss(para_w, para_bias, d_train[:,0], d_train[:,1]))
    
np.save("trained_w.npy", para_w)
np.save("trained_b.npy", para_bias)
np.save("data_mean.npy", d_mean)
np.save("data_std.npy", d_std)

#checking
with open("record.csv", "a") as f:
    final_loss = loss(para_w, para_bias, validation_array[:,0], validation_array[:,1]), loss(para_w, para_bias, training_array[:,0], training_array[:,1])
    print("%f,%f"%(final_loss[0], final_loss[1]), file = f)
 
               

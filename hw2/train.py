# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:18:09 2018

@author: Austin Hsu
"""

####################
##                ##
##Generative Model##
##                ##
####################

import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)

x_train = pd.read_csv('train_x.csv', engine = 'python')
x_train = x_train.values

y_train = pd.read_csv('train_y.csv', engine = 'python')
y_train = y_train.values

#mu_star = np.array([[np.mean(x_train[:,i])] for i in range(x_train.shape[1])])
#sigma_star = 1/x_train.shape[0]*np.array([np.dot(x_train[i] - mu_star,(x_train[i] - mu_star).T) for i in range(x_train.shape[1])]).sum(axis = 0)

x_train_0 = np.array([x_train[i] for i in range(x_train.shape[0]) if y_train[i] == 0])
x_train_1 = np.array([x_train[i] for i in range(x_train.shape[0]) if y_train[i] == 1])

mu_0 = np.array([[np.mean(x_train_0[:,i])] for i in range(x_train_0.shape[1])])
mu_1 = np.array([[np.mean(x_train_1[:,i])] for i in range(x_train_1.shape[1])])

sigma_0 = sum([np.dot(x_train_0[i].reshape((23,1)) - mu_0,(x_train_0[i].reshape((23,1)) - mu_0).T) for i in range(x_train_0.shape[0])])
sigma_1 = sum([np.dot(x_train_1[i].reshape((23,1)) - mu_1,(x_train_1[i].reshape((23,1)) - mu_1).T) for i in range(x_train_1.shape[0])])
sigma_total = sigma_0 / x_train.shape[0] + sigma_1 / x_train.shape[0]
sigma_inv = np.linalg.inv(sigma_total)

w = np.dot((mu_0-mu_1).T, sigma_inv)
b = -0.5*np.dot(mu_0.T, np.dot(sigma_inv, mu_0)) + 0.5*np.dot(mu_1.T, np.dot(sigma_inv, mu_1)) + np.log(x_train_0.shape[0]/x_train_1.shape[0])

np.save("generative_w.npy", w)
np.save("generative_b.npy", b)
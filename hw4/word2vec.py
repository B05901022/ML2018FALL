#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 19:08:39 2018

@author: austinhsu
"""

import jieba
import numpy as np

jieba.load_userdict('dict.txt.big')

input_datas = [list(jieba.cut(i.split(',')[1])) for i in open('./train_x.csv', 'r').read().split('\n')[1:-1]]#pd.read_csv('./train_x.csv', encoding = 'utf8')
input_label = [int(i.split(',')[1]) for i in open('train_y.csv', 'r').read().split('\n')[1:-1]]

#117174 words in total

from gensim.models import word2vec

model = word2vec.Word2Vec(input_datas, size=250, window=5, min_count=5, workers=4, iter=10, sg=1)

model.save("dcard_word2vec.model")

np.save("dcard_labels.npy", np.array(input_label))
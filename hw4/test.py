#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 22:10:36 2018

@author: austinhsu
"""

import numpy as np
import jieba
from gensim.models import word2vec
#from keras.utils import to_categorical
#from keras.preprocessing.sequence import pad_sequences
#from keras.models import Sequential
#from keras.layers import Embedding, CuDNNGRU, Dense, TimeDistributed, BatchNormalization, LeakyReLU, Dropout, LSTM, GRU, Bidirectional

jieba.load_userdict('dict.txt.big')
w2v_model = word2vec.Word2Vec.load("dcard_word2vec.model")

#embedding layer
embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
word2idx = {}

vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1

"""    
embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False)
"""

def text_to_index(corpus):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)

#cut the sentence
input_datas = [list(jieba.cut(i.split(',')[1])) for i in open('./test_x.csv', 'r').read().split('\n')[1:-1]]

#load and generate train data/label
#label = to_categorical(np.load("dcard_labels.npy")).reshape(120000,1,2)#np.load("dcard_labels.npy").reshape(120000,1,1)#
test_data = text_to_index(input_datas)

from keras.preprocessing.sequence import pad_sequences
#padding
padding_length = 200
test_data = pad_sequences(test_data, maxlen=padding_length)


from keras.models import Sequential, load_model

model = Sequential()
model = load_model("hw4_model.h5")
prediction = model.predict_classes(test_data)

with open("result.csv", 'w') as f:
    print('id,label', file = f)
    for i in range(prediction.shape[0]):
        print('%d,%d' % (i,prediction[i][0]), file = f)


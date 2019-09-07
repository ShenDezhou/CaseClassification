#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2019年9月7日

@author: Administrator
'''
from __future__ import print_function

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import os
import codecs
from sklearn.metrics import accuracy_score
import numpy
from sklearn.model_selection import train_test_split
import pkuseg
import datetime
import pickle
from functools import partial
import math

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical

# data_dir = 'data/Artistes_et_Phalanges-David_Campion'  # data directory containing input.txt
# save_dir = 'save'  # directory to store models
rnn_size = 128  # size of RNN
batch_size = 30  # minibatch size
seq_length = 15  # sequence length
num_epochs = 8  # number of epochs
learning_rate = 0.001  # learning rate
sequences_step = 1  # step to create sequences

# create sequences
# sequences = []
# next_words = []
# for i in range(0, len(x_text) - seq_length, sequences_step):
#     sequences.append(x_text[i: i + seq_length])
#     next_words.append(x_text[i + seq_length])
# 
# print('nb sequences:', len(sequences))
# 

# step = 4
# 
# 
# def convertToMatrix(matrix, step):
#     X, Y = [], []
#     for i in range(matrix.shape[1] - step):
#         d = i + step  
#         X.append(matrix[:, i:d])
#         Y.append(matrix[:, d])
#     return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
# 
# 
# trainX, trainY = convertToMatrix(train, step)
# testX, testY = convertToMatrix(test, step)

train_percent = 0.8
STATE = 1234

for filename in os.listdir(u"./cases"):
    if filename == 'execution.txt':
        continue
    data = []
    data_labels = []
    
    if filename.endswith(".txt"):
        token = filename.replace(".txt", "")
        
        with codecs.open("./cases/" + filename, 'r', encoding='utf-8') as f:
            for text in f:
                xy = text.split('|')
                if len(xy) > 1:
                    data.append(xy[1])
                    data_labels.append(xy[0])
        
        print(len(data), len(data_labels))
        
        features_nd = numpy.load("./numpy/" + token + ".npz")["nd"]
        print("3:" + token + ".npz")
        with open("./tags/" + token + "-y.pkl", "rb") as f:
            s = f.read()
            data_labels = pickle.loads(s)
            print("3-1 tags pickle:", len(s))
        
        lbe = LabelEncoder()
        data_labels = lbe.fit_transform(data_labels).tolist()
        data_labels = to_categorical(data_labels, dtype=int)
        print("3-2 label encode:", len(data_labels))
        
        # reshape for rnn
        features_nd = features_nd.reshape((features_nd.shape[0], features_nd.shape[1], 1))
        
        X_train, X_test, y_train, y_test = train_test_split(features_nd,
                                                            data_labels,
                                                            train_size=train_percent,
                                                            random_state=STATE)
        print("4:split", train_percent)
        
        before_training = datetime.datetime.now()
        # build the model: a single LSTM
        print('5:Build LSTM model.')
        model = Sequential()
        model.add(LSTM(rnn_size, input_shape=(features_nd.shape[1], features_nd.shape[2])))
        model.add(Dense(data_labels.shape[1]))
        model.add(Activation('softmax'))
        
        # adam optimizer
        optimizer = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        
        # fit the model
        model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs)
        after_training = datetime.datetime.now()
        
        with open("./model/" + token + "rnn.pkl", "wb") as f:
            s = pickle.dumps(model)
            f.write(s)
            print("5-2 pickle:", len(s))
          
        train_pred = model.predict(X_train)
        print(token, '@train-accuracy-score', accuracy_score(y_train, train_pred))
        print("5:training time(sec):", str((after_training - before_training).total_seconds()))
         
        y_pred = model.predict(X_test)
        print(token, '@test-score', accuracy_score(y_test, y_pred))
        print("6:test")

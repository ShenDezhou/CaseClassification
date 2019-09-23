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
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
# data_dir = 'data/Artistes_et_Phalanges-David_Campion'  # data directory containing input.txt
# save_dir = 'save'  # directory to store models
rnn_size = 128  # size of RNN
batch_size = 30  # minibatch size
seq_length = 15  # sequence length
num_epochs = 8  # number of epochs
learning_rate = 0.01  # learning rate
sequences_step = 1  # step to create sequences

max_features = 20000
maxlen = 20000
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 200

train_percent = 0.8
STATE = 1234

for filename in os.listdir(u"./cases"):
    if filename != 'execution.txt':
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
        print("2.shape:", features_nd.shape)
        features_nd = sequence.pad_sequences(features_nd, maxlen=maxlen)
        print("2.1 shape:", features_nd.shape)
        
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
#         features_nd = features_nd.reshape((features_nd.shape[0], features_nd.shape[1], 1))
        
        X_train, X_test, y_train, y_test = train_test_split(features_nd, \
                                                            data_labels, \
                                                            train_size=train_percent, \
                                                            random_state=STATE)
        print("4:split", train_percent)
        
        before_training = datetime.datetime.now()
        # build the model: a single LSTM
        print('5:Build cnn model.')
        model = Sequential()
        model.add(Embedding(max_features, \
                    embedding_dims, \
                    input_length=maxlen))
        model.add(Dropout(0.2))

        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        model.add(Conv1D(filters, \
                         kernel_size, \
                         padding='valid', \
                         activation='relu', \
                         strides=1))
        # we use max pooling:
        model.add(GlobalMaxPooling1D())
        
        # We add a vanilla hidden layer:
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        
        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(8))
        model.add(Activation('sigmoid'))
         
        # adam optimizer
        optimizer = Adam(lr=learning_rate)
        
        model.compile(loss='categorical_crossentropy', \
                      optimizer=optimizer, \
                      metrics=['accuracy'])
        
        earlystop = [EarlyStopping(monitor='loss', min_delta=1e-6, patience=3, verbose=0, mode='min')]
        
        model.fit(X_train, y_train, \
                  batch_size=batch_size, \
                  epochs=epochs, \
                  callbacks=earlystop, \
                  validation_data=(X_test, y_test))
       
        # fit the model
        # model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs)
        after_training = datetime.datetime.now()
        
        with open("./model/" + token + "cnn.pkl", "wb") as f:
            s = pickle.dumps(model)
            f.write(s)
            print("5-2 pickle:", len(s))
        
        print(token, '@train-accuracy-score', model.evaluate(X_train, y_train,
                            batch_size=batch_size))
        print("5:training time(sec):", str((after_training - before_training).total_seconds()))
         
        print(token, '@test-score', model.evaluate(X_test, y_test,
                            batch_size=batch_size))
        print("6:test")

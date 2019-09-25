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
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten

from keras.optimizers import Adam
from keras.utils import to_categorical
from numpy import argmax

from keras.callbacks import EarlyStopping
from keras.models import load_model

# data_dir = 'data/Artistes_et_Phalanges-David_Campion'  # data directory containing input.txt
# save_dir = 'save'  # directory to store models
rnn_size = 128  # size of RNN
batch_size = 30  # minibatch size
seq_length = 15  # sequence length
num_epochs = 50  # number of epochs
learning_rate = 0.001  # learning rate
sequences_step = 1  # step to create sequences

ngram_range = 1
max_features = 20000
maxlen = 20000
batch_size = 1
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 200

train_percent = 0.99
STATE = 1234

for filename in os.listdir(u"./cases"):
    if filename != 'accuse.txt':
        continue
    data = []
    data_labels = []
    
    if filename.endswith(".txt"):
        token = filename.replace(".txt", "")
        
#         with codecs.open("./cases/" + filename, 'r', encoding='utf-8') as f:
#             for text in f:
#                 xy = text.split('|')
#                 if len(xy) > 1:
#                     data.append(xy[1])
#                     data_labels.append(xy[0])
#         print(len(data), len(data_labels))
        
        if token in ["civil", "criminal", "accuse"]:
            with open("./numpy/" + token + "csr.pkl", "rb") as f:
                s = f.read()
                features_nd = pickle.loads(s)
                print("3 numpy pickle:", len(s))
        else:
            features_nd = numpy.load("./numpy/" + token + ".npz")["nd"]
            print("3:" + token + ".npz")
            
        with open("./tags/" + token + "-y.pkl", "rb") as f:
            s = f.read()
            data_labels = pickle.loads(s)
            print("3-1 tags pickle:", len(s))
        
        lbe = LabelEncoder()
        data_labels = lbe.fit_transform(data_labels)
        with open("./model/" + token + "labelencoder.pkl", "wb") as f:
            s = pickle.dumps(lbe)
            f.write(s)
            print("3-2 label encoder pickle:", len(s))
            
        data_labels = to_categorical(data_labels, dtype=int)
        print("3-3 label encode:", len(data_labels))

        ratio = int(features_nd.shape[0] * train_percent)  # should be int
        X_train = features_nd[:ratio, :]
        X_test = features_nd[ratio:, :]
        y_train = data_labels[:ratio, :]
        y_test = data_labels[ratio:, :]
#         X_train, X_test, y_train, y_test = train_test_split(features_nd,
#                                                             data_labels,
#                                                             train_size=train_percent,
#                                                             random_state=STATE)
        print("4:split", train_percent)
        
        before_training = datetime.datetime.now()
        # build the model: a single GAP
        print('5:Build GAP model.')
        model = Sequential()
        model.add(Embedding(X_train.shape[1], \
                    embedding_dims, \
                    input_length=X_train.shape[1]))
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
        model.add(Dense(data_labels.shape[1]))
        model.add(Activation('sigmoid'))
              
        # adam optimizer
        optimizer = Adam(lr=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        earlystop = [EarlyStopping(monitor='loss', min_delta=5e-3, patience=1, verbose=1, mode='min')]
        #print model summary
        print("summary",model.summary())
        # fit the model
        model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=earlystop)
        after_training = datetime.datetime.now()
        
        with open("./model/" + token + "gapnetwork.pkl", "wb") as f:
            s = pickle.dumps(model.to_json())
            f.write(s)
            print("5-2 model network pickle:", len(s))
            
        with open("./model/" + token + "gapweights.pkl", "wb") as f:
            s = pickle.dumps(model.get_weights())
            f.write(s)
            print("5-3 weight pickle:", len(s))
        
          
        train_pred = model.predict(X_train, batch_size=batch_size)
        print(token, '@train-accuracy-score', model.evaluate(X_train, y_train, batch_size=batch_size))
        print("5:training time(sec):", str((after_training - before_training).total_seconds()))
#         print("5.1:decode expect", lbe.inverse_transform(numpy.argmax(y_train, axis=1)), ",actual:", lbe.inverse_transform(argmax(train_pred, axis=1))) 
        
        test_pred = model.predict(X_test, batch_size=batch_size)
        print(token, '@test-score', model.evaluate(X_test, y_test, batch_size=batch_size))
        print("6:test")
#         print("6.1:decode expect", lbe.inverse_transform(argmax(y_test, axis=1)), ",actual:", lbe.inverse_transform(argmax(test_pred, axis=1)))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2019年5月17日

@author: Administrator
'''
from sklearn.feature_extraction.text import CountVectorizer
import os
import codecs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy
from sklearn.model_selection import train_test_split
import pkuseg
import datetime
import pickle
from sklearn_porter import Porter


seg = pkuseg.pkuseg(model_name='web')
train_percent=0.8
STATE=1234

before_model = datetime.datetime.now()
        
for filename in os.listdir(u"./cases"):
    data = []
    data_labels = []
    log_model = LogisticRegression()
    if filename.endswith(".txt"):
        token = filename.replace(".txt","")
        with codecs.open("./cases/"+filename, 'r', encoding='utf-8') as f:
            for text in f:
                xy = text.split('|')
                data.append(xy[0])
    #             try:
    #                 t=int(xy[1])
    #             except ValueError:
    #                 print("1:",xy[0])
    #                 break
                data_labels.append(xy[1])
        
        print(len(data), len(data_labels))
#         if len(data)%100!=0:
#             print("2:",token)
#             break
        
        
        vectorizer = CountVectorizer(
            analyzer = lambda text: seg.cut(text),
            lowercase = False,
        )
        features = vectorizer.fit_transform(
            data
        )
        features_nd = features.toarray()
        numpy.savez_compressed("./numpy/"+token+".npz", nd=features_nd)
        print("3:"+token+".npz")
        
        X_train, X_test, y_train, y_test  = train_test_split(
        features_nd, 
        data_labels,
        train_size=train_percent, 
        random_state=STATE)
        print("4:split", train_percent)
        
        before_training = datetime.datetime.now()
        log_model = log_model.fit(X=X_train, y=y_train)
        after_training = datetime.datetime.now()
        
        with open("./model/"+token+".pkl", "wb") as f:
            s = pickle.dumps(log_model)
            f.write(s)
            print("5-1 pickle:", len(s))
        
#         with open("./model/"+token+".java", "w") as f:
#             porter = Porter(log_model, language='java')
#             s = porter.export(embed_data=True)
#             f.write(s)
#             print("5-2 java:", len(s))
        
        train_pred = log_model.predict(X_train)
        print(token,'@train-score', accuracy_score(y_train, train_pred))
        print("5:training time(sec):", str((after_training-before_training).total_seconds()))
        
        y_pred = log_model.predict(X_test)
        print(token,'@test-score', accuracy_score(y_test, y_pred))
        print("6:test")
        continue
    else:
        continue

after_model = datetime.datetime.now()
print("ALL-DONE", str((after_model-before_model).total_seconds()))
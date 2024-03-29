#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2019年9月7日

@author: Administrator
'''
from sklearn.feature_extraction.text import CountVectorizer
import os
import codecs
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy
from sklearn.model_selection import train_test_split
import pkuseg
import datetime
import pickle
from functools import partial
from pathlib import Path
import argparse


def cut(text, aseg):
    return aseg.cut(text)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--category', choices=["execution","civil","criminal", "statecompensation","administrative","accuse"],
                        help='cases category:"execution","civil","criminal", "statecompensation","administrative","accuse"')
    args = parser.parse_args()
    
    token = args.category
    sample=[]
    sample_labels=[]
    i=0
    with codecs.open("./cases/" + token + ".txt", 'r', encoding='utf-8') as f:
        for text in f:
            xy = text.split('|')
            if len(xy) > 1:
                sample.append(xy[1])
                sample_labels.append(xy[0])
            i+=1
            if i>10:
                break
        
    with open("./model/" + token + "countvectorizer.pkl", "rb") as f:
        s = f.read()
        print("1:vectorizer", len(s))
        vectorizer = pickle.loads(s)
        
    
    features = vectorizer.transform(
            sample
            )
    features_nd = features.toarray()
    
    with open("./model/" + token + "decisiontree.pkl", "rb") as f:
        s = f.read()
        cart_model = pickle.loads(s)
        print("2:decisiontree", len(s))
    
    sample_pred = cart_model.predict(features_nd)
    print("3:", token, "expected:", sample_labels, "actual:", sample_pred)
    
    print("ALL-DONE")
    

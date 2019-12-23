#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 10:23:58 2019

@author: caglar
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#Load delected dataset
def load_dataset(dataset):
    global data
    data = pd.read_csv('datasets/' + dataset)
    return list(data.columns)


# Choose the target column
def init_dataset(target):
    global data, x, y, X, Y
    y = pd.DataFrame(data[target])
    Y = data[target].values
    x = data.drop([target],axis=1)
    

def gradient_descent_g():
    '''
    return: ROC function
    
    '''
    global params_optimal
    
    m = len(y)
    X = np.hstack((np.ones((m,1)),x))
    n = np.size(X,1)
    params = np.zeros((n,1))
    #print(y)
    iterations = 1500
    learning_rate = 0.03
    (cost_history, params_optimal) = gradient_descent(X, y, params, learning_rate, iterations)
    y_pred = predict(X,params_optimal)
    print(type(Y))
    print(type(y_pred))
    return ROC(Y,y_pred.to_numpy())
    
def gradient_descent(X, y, params, learning_rate, iterations):
    '''
    @parameters:
        X: features
        y: label
        params: weights
    Calculate the derivative of cost function and update the params
    '''
    m = len(y)
    cost_history = np.zeros((iterations,1))
    
    for i in range(iterations):
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y)) 

    return (cost_history, params)

# Predict with X and weights
def predict(X, params):
    return sigmoid(X @ params)


# Calculate to Draw ROC
def ROC(y,y_pred):
   
    # false positive rate
    fpr = []
    # true positive rate
    tpr = []
    # Iterate thresholds from 0.0, 0.01, ... 1.0
    thresholds = np.arange(0.0, 1.01, .01)
    
    # get number of positive and negative examples in the dataset
    P = sum(y)
    N = len(y) - P
    
    # iterate through all thresholds and determine fraction of true positives
    # and false positives found at this threshold
    for thresh in thresholds:
        FP=0
        TP=0
        for i in range(len(y_pred)):
            if (y_pred[i] > thresh):
                if y[i] == 1:
                    TP = TP + 1
                if y[i] == 0:
                    FP = FP + 1
        fpr.append(FP/float(N))
        tpr.append(TP/float(P))
    return fpr,tpr

def roc_g():
    return ROC(y,predict(X,params_optimal))

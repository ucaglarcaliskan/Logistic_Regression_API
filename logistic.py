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


#data = pd.read_csv('bloodpress.csv',header=None,names=['age','pressure'])
#data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
#data = pd.read_csv('seeds_dataset.txt', header=None, sep='\t',error_bad_lines=False)
# =============================================================================
# data = pd.read_csv('sonar.csv', header=None)
# #data = pd.read_csv('cdataset.csv')
# y = data.iloc[:,-1:].values
# x = data.iloc[:,:-1].values
# 
# =============================================================================
#X = np.hstack((np.ones((m,1)),x))



# =============================================================================
# x = data['age']
# y = data['pressure']
# plt.plot(x,y,'b.')
# plt.show()
# =============================================================================
# =============================================================================
# x = np.array([[1,2,3,4,5],[2,4,6,8,11]])
# x = x.T
# y = np.array([0,0,0,0,1])
# y = y.reshape(-1,1)
# 
# =============================================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# =============================================================================
# def compute_cost(X, y, theta):
#     m = len(y)
#     h = sigmoid(X @ theta)
#     epsilon = 1e-5
#     cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
#     return cost
# =============================================================================

def load_dataset(dataset):
    global data
    data = pd.read_csv('datasets/' + dataset)
    return list(data.columns)

def init_dataset(target):
    global data, x, y, X, Y
    y = pd.DataFrame(data[target])
    Y = data[target].values
    x = data.drop([target],axis=1)
    
    #X = np.hstack((np.ones((m,1)),x))
    
def gradient_descent_g():
    global params_optimal
    
    m = len(y)
    X = np.hstack((np.ones((m,1)),x))
    n = np.size(X,1)
    params = np.zeros((n,1))
    #print(y)
    iterations = 1500
    learning_rate = 0.03
    (cost_history, params_optimal) = gradient_descent(X, y, params, learning_rate, iterations)
    #my_list = map(lambda x: x[0], cost_history)
    #cost_history = pd.Series(my_list).to_numpy() #Array of array problem is solved
    #print(cost_history)
    #cost_history = str(cost_history)
    #params_optimal = np.array_str(params_optimal)
    #return (cost_history, params_optimal)
    y_pred = predict(X,params_optimal)
    print(type(Y))
    print(type(y_pred))
    return ROC(Y,y_pred.to_numpy())
    
def gradient_descent(X, y, params, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros((iterations,1))
    #print(params.shape)
    #print(X.T.shape)
    #print(y.shape)
    for i in range(iterations):
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y)) 
        #cost_history[i] = compute_cost(X, y, params)

    return (cost_history, params)

def predict(X, params):
    return sigmoid(X @ params)



# =============================================================================
# initial_cost = compute_cost(X, y, params)
# 
# print("Initial Cost is: {} \n".format(initial_cost))
# 
# (cost_history, params_optimal) = gradient_descent(np.hstack((np.ones((m,1)),x)), y, , learning_rate, iterations)
# 
# print("Optimal Parameters are: \n", params_optimal, "\n")
# 
# plt.figure()
# sns.set_style('white')
# plt.plot(range(len(cost_history)), cost_history, 'r')
# plt.title("Convergence Graph of Cost Function")
# plt.xlabel("Number of Iterations")
# plt.ylabel("Cost")
# plt.show()
# 
# =============================================================================
#y_pred = predict(X,params_optimal)

import collections
ConfusionMatrix = collections.namedtuple('conf', ['tp','fp','tn','fn']) 

def calc_ConfusionMatrix(actuals, scores, threshold=0.5, positive_label=1):
    tp=fp=tn=fn=0
    bool_actuals = [act==positive_label for act in actuals]
    for truth, score in zip(bool_actuals, scores):
        if score > threshold:                      # predicted positive 
            if truth:                              # actually positive 
                tp += 1
            else:                                  # actually negative              
                fp += 1          
        else:                                      # predicted negative 
            if not truth:                          # actually negative 
                tn += 1                          
            else:                                  # actually positive 
                fn += 1
    return ConfusionMatrix(tp, fp, tn, fn)




#conf = calc_ConfusionMatrix(y,y_pred)
def ROC(y,y_pred):
    #y = y.reshape(1,-1)
    #y_pred = y_pred.reshape(1,-1)
# =============================================================================
#     my_list = map(lambda x: x[0], y_pred)
#     y_pred = pd.Series(my_list) #Array of array problem is solved
# =============================================================================
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
# =============================================================================
#     plt.scatter(fpr, tpr, s=1, color='orange')
#     plt.show()
# =============================================================================
    return fpr,tpr

def roc_g():
    return ROC(y,predict(X,params_optimal))

#roc_g()

# =============================================================================
# def FPR(conf_mtrx):
#     return conf_mtrx.fp / (conf_mtrx.fp + conf_mtrx.tn) if (conf_mtrx.fp + conf_mtrx.tn)!=0 else 0
# def TPR(conf_mtrx):
#     return conf_mtrx.tp / (conf_mtrx.tp + conf_mtrx.fn) if (conf_mtrx.tp + conf_mtrx.fn)!=0 else 0
# 
# fpr = FPR(conf)
# tpr = TPR(conf)
# 
# def apply(actuals, scores, **fxns):
#     # generate thresholds over score domain 
#     low = min(scores)
#     high = max(scores)
#     step = (abs(low) + abs(high)) / 1000
#     thresholds = np.arange(low-step, high+step, step)
#     # calculate confusion matrices for all thresholds
#     confusionMatrices = []
#     for threshold in thresholds:
#         confusionMatrices.append(calc_ConfusionMatrix(actuals, scores, threshold))
#     # apply functions to confusion matrices 
#     results = {fname:list(map(fxn, confusionMatrices)) for fname, fxn in fxns.items()}
#     results["THR"] = thresholds
#     return results
# def ROC(y_true, y_pred):
#     return apply(y_true,  y_pred, FPR=fpr,TPR=tpr)
# 
# #a = ROC(y,y_pred)
# plt.plot(fpr,tpr)
# plt.show() 
# 
# auc = np.trapz(y,x)
# =============================================================================

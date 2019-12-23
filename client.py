#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 17:17:11 2019

@author: caglar
"""

import requests
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ntpath

ntpath.basename("a/b/c")
server_url =  'http://127.0.0.1:5000'
datasets = []

def configure():
    print('\n\nCurrent URL: ' + server_url)
    address = input('\n\nEnter new URL (0 for discard): ')
    if address.split() == '0':
        pass
    else:
        server_url = address.split()
def train():
    list_datasets()
    selection = input('\nEnter the dataset index: ')
    select_dataset(int(selection))
    
    response = requests.get(server_url + '/train')
    res_json = response.json()
    
    fpr = np.asarray(res_json['fpr'])
    tpr = np.asarray(res_json['tpr'])
    draw_roc(fpr,tpr)

def draw_roc(fpr,tpr):
    plt.scatter(fpr, tpr, s=1, color='orange')
    plt.show()

def select_dataset(selection):
    global datasets
    selected = datasets[selection]
    print(selected)
    response = requests.get(server_url + '/dataset/'+ selected)
    res_json = response.json()
    columns = res_json['columns']
    for i in columns:
        print(i)
    selection = input('\nEnter the exact name of label column: ')
    response = requests.get(server_url + '/label/'+ selection)

    
def upload_dataset():
    path = input('Exact path of dataset(e.g.: ./datasets/dataset.csv): ')
    url = server_url + '/uploader/' + path_leaf(path)
    fin = open(path, 'rb')
    files = {'file': fin}
    try:
        r = requests.post(url, files=files)
        print(r.text)
    finally:
    	fin.close()
        
def list_datasets():
    global datasets
    response = requests.get(server_url + '/list')
    res_json = response.json()
    datasets = res_json['datasets']
    print('\n\nDATASETS\n')
    for i in enumerate(datasets):
        print(i)
 
#Get filename
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
#upload_dataset()
#train()
#list_datasets()
while(True):
    print('\n\n\n1 - Configure Server Information')
    print('2 - List All Dataset')
    print('3 - Upload New Dataset')
    print('4 - Train a Model')
    sel = input('\nSelect the option: ')
    
    if sel=='1':
        pass
    elif sel=='2':
        list_datasets()
    elif sel=='3':
        upload_dataset()
    elif sel=='4':
        train()
    
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:27:32 2020

@author: vincenzodigennaro
"""

import time
from IPython.display import display, clear_output
from scipy.special import expit
from scipy.stats import truncnorm
import numpy as np
import pandas as pd

########## useful functions #############
def ReLU(x):
    return np.where(x>0.0,x,0.0)

def ReLU_derivation(x):
    return np.where(x>0.0,1.0,0.0)
    
def LeakyReLU(x, slope=0.01):
    return np.where(x>0.0, x, x*slope)

def LReLU_deriv(x, slope=0.01):
    return np.where(x>0.0,1.0,slope)
    
def softmax(x):
    exp_x = np.exp(x - x.max())
    return exp_x / exp_x.sum(axis=0, keepdims=True)
    
def sigmoid(x):
    return expit(x)

def sigmoid_der(x):
    return expit(x)*(1 - expit(x))

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def encode_labels(y, k):
    y = y.astype(int)
    onehot = np.zeros((k, y.shape[0]))
    for idx, val in enumerate(y):
        onehot[val, idx] = 1.0
    return onehot

from sklearn.metrics import confusion_matrix
def plot_conf_matrix(true_y, pred, normalized=True, labels=None):
    true_y = true_y.reshape(-1,1)
    pred = pred.reshape(-1,1)
    score = (pred==true_y).mean()
    cm = confusion_matrix(true_y, pred)
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if labels!=None:
        cm = pd.DataFrame(cm, index=labels_emotions, columns=labels_emotions)
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'coolwarm');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {:.3f}'.format(score) 
    plt.title(all_sample_title, size = 15);
    plt.subplot(1,2,2)
    sns.heatmap(cm-np.diag(np.diag(cm)), annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'coolwarm');    
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {:.3f}'.format(score) 
    plt.title(all_sample_title, size = 15);


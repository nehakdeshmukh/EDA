# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:29:36 2023

@author: nehak
"""


import pandas as pd 
from sklearn.datasets import make_blobs
from sklearn import datasets
import numpy as np


class PCA:
  def __init__(self, n_components):
    self.n_components = n_components
    self.components = None
    self.mean = None
    
  def fit(self, X):
    self.mean = np.mean(X, axis=0)
    X = X - self.mean
    cov = np.cov(X.T)
    
    evalue, evector = np.linalg.eig(cov)
    
    eigenvectors = evector.T
    idxs = np.argsort(evalue)[::-1]
    
    evalue = evalue[idxs]
    evector = evector[idxs]
    self.components = evector[0:self.n_components]
    
  def transform(self, X):
    #project data
    X = X - self.mean
    return(np.dot(X, self.components.T))


data = datasets.load_iris()
X = data.data
y = data.target

pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

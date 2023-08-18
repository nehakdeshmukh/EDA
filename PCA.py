# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:29:36 2023

@author: nehak
"""


import pandas as pd 
from sklearn.datasets import make_blobs
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


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


x1 = X_projected[:,0]
x2 = X_projected[:,1]

plt.scatter(x1,x2,c=y,edgecolor='none',alpha=0.8,cmap=plt.cm.get_cmap('viridis',3))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()


# using module 
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
 
X_projected_pca = pca.fit_transform(X)

plt.scatter(X_projected_pca[:,0],X_projected_pca[:,1],c=y,edgecolor='none',alpha=0.8,cmap=plt.cm.get_cmap('viridis',3))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()

print("explained variance: " ,pca.explained_variance_ratio_)

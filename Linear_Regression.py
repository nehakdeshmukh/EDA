# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:42:08 2023

@author: nehak
"""

# Linear Regression 


import numpy as np
import pandas as pd 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


data_train = pd.read_csv(r"C:\Users\nehak\Downloads\train.csv")

data_test = pd.read_csv(r"C:\Users\nehak\Downloads\test.csv")


data_train.info()
data_test.info()


data_train.describe()
data_test.describe()

data_train= data_train.dropna()
data_test = data_test.dropna()


X_train = np.array(data_train.iloc[:, :-1].values)
y_train = np.array(data_train.iloc[:, 1].values)
X_test = np.array(data_test.iloc[:, :-1].values)
y_test = np.array(data_test.iloc[:, 1].values)
model = LinearRegression()


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(data_test["y"], y_pred)
MSE = mean_squared_error(data_test["y"], y_pred)







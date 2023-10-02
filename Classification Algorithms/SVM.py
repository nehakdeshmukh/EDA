# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:29:20 2023

@author: nehak
"""

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

data_svm = pd.read_csv(r"C:\Users\nehak\Downloads\UniversalBank.csv")

data_svm.info()

data_svm.describe()

data_x = data_svm.iloc[:,:13]
data_y = data_svm.iloc[:,-1:]

scaler = StandardScaler()
scale_x = scaler.fit_transform(data_x)


train_x, test_x, train_y, test_y = train_test_split(scale_x,data_y,test_size=0.2)

model = SVC(kernel='rbf',random_state=5)
model.fit(train_x, train_y)

prediction = model.predict(test_x)

acc = accuracy_score(test_y, prediction)



# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:29:20 2023

@author: nehak
"""

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

data_svm = pd.read_csv(r"C:\Users\nehak\Downloads\UniversalBank.csv")

data_svm.info()

data_svm.describe()

data_x = data_svm.iloc[:,:13]
data_y = data_svm.iloc[:,-1:]



train_x, test_x, train_y, test_y = train_test_split(data_x,data_y,test_size=0.2)

model = SVC()
model.fit(train_x, train_y)

prediction = model.predict(test_x)

acc = accuracy_score(test_y, prediction)



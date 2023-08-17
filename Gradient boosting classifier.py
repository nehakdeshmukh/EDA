# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:37:02 2023

@author: nehak
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import time


data = pd.read_csv(r"C:\Users\nehak\Downloads\enhanced_tips_dataset.csv")

data.describe()

data.info()

data.isnull().sum()

data = data.drop(['sex','id'],axis=1)
# num_col = data.select_dtype(['int64'],'float64')

data_x = data.loc[:, data.columns != 'sex_binary']
data_y = data.loc[:, 'sex_binary']


num_col = data_x.select_dtypes(['int64', 'float64'])
cat_col = data_x.select_dtypes('object')

encoder = LabelEncoder()
cat_col = cat_col.apply(encoder.fit_transform)

data_x_updated = pd.DataFrame(pd.concat([num_col, cat_col], axis=1))

train_x, test_x, train_y, test_y = train_test_split(
    data_x_updated, data_y, test_size=0.2)

gbk = GradientBoostingClassifier()
gbk.fit(train_x, train_y)
pred = gbk.predict(test_x)
acc=accuracy_score(test_y, pred)

# Hyper parameter tunning 

GBC_param={
    "loss":['log_loss', 'deviance', 'exponential'],
    "learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
    "n_estimators":[10,20,30,40,50,60,70,80,90],
    "max_depth":[2,3,4,5],
    "min_samples_split":[2,3,4,5]
                 }

GV_model = GridSearchCV(gbk, GBC_param, cv=5, verbose=3)


s_time = time.time()
GV_model.fit(train_x, train_y)
e_time = time.time()

print("total time :", e_time-s_time)



# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:04:18 2023

@author: nehak
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import time
from sklearn.ensemble import RandomForestClassifier

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

train_x, test_x, train_y, test_y = train_test_split(data_x_updated, data_y, 
                                                    test_size=0.2)

## Random Forest 

RF_model = RandomForestClassifier()
RF_model.fit(train_x,train_y)
RF_prediction = RF_model.predict(test_x)
RF_accuracy = accuracy_score(test_y,RF_prediction)

CM = confusion_matrix(test_y,RF_prediction)
# Hyper parameter tunning 

RF_param={
    "n_estimators":[10, 100,1000],
    "criterion":["gini", "entropy", "log_loss"],
    "max_features":['sqrt', 'log2'],
    "max_depth":[2,3,4,5],
    "min_samples_split":[2,3,4,5]
                 }
    

GV_model = GridSearchCV(RF_model, RF_param, cv=5, verbose=3)

s_time = time.time()
GV_model.fit(train_x, train_y)
e_time = time.time()

print("total time :", e_time-s_time)

GV_model.best_params_

# Tunned RF model 

TRF_model = RandomForestClassifier(criterion='gini', max_depth=2,
                                   max_features='sqrt',min_samples_split=2,
                                   n_estimators=10)
TRF_model.fit(train_x,train_y)
TRF_prediction = TRF_model.predict(test_x)
TRF_accuracy = accuracy_score(test_y,TRF_prediction)

CM = confusion_matrix(test_y,TRF_prediction)
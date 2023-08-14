# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 09:53:37 2023

@author: nehak
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR 
from sklearn.model_selection import cross_val_score, GridSearchCV

# Regression algo

data = pd.read_csv(r"C:\Users\nehak\Downloads\enhanced_tips_dataset.csv")

data.describe()

data.info()

data.isnull().sum()


# num_col = data.select_dtype(['int64'],'float64')

data_x = data.loc[:, data.columns != 'tip_percentage']
data_y = data.loc[:, 'tip_percentage']


num_col = data_x.select_dtypes(['int64', 'float64'])
cat_col = data_x.select_dtypes('object')

encoder = LabelEncoder()
cat_col = cat_col.apply(encoder.fit_transform)

data_x_updated = pd.DataFrame(pd.concat([num_col, cat_col], axis=1))

train_x, test_x, train_y, test_y = train_test_split(
    data_x_updated, data_y, test_size=0.2)



# SVM Regressor 

SVM_model = SVR()
SVM_model.fit(train_x, train_y)
SVM_Prediction = SVM_model.predict(test_x)
SVM_MSE = mean_squared_error(test_y, SVM_Prediction)


print('SVM_MAE:', mean_absolute_error(test_y, SVM_Prediction))
print('SVM_MSE:', mean_squared_error(test_y, SVM_Prediction))
print('SVM_RMSE:', np.sqrt(mean_squared_error(test_y, SVM_Prediction)))
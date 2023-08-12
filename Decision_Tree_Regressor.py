# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 13:06:57 2023

@author: nehak
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV


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

# decision tree Regressor

DT_model = DecisionTreeRegressor()
DT_model.fit(train_x, train_y)
DT_Prediction = DT_model.predict(test_x)
DT_MSE = mean_squared_error(test_y, DT_Prediction)

print('MAE:', mean_absolute_error(test_y, DT_Prediction))
print('MSE:', mean_squared_error(test_y, DT_Prediction))
print('RMSE:', np.sqrt(mean_squared_error(test_y, DT_Prediction)))


# Hyper parameter tunning 

param_grid = {"splitter":["best", "random"],
              "max_depth":[2,3,4,5,6,7,8,9,10],
              "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
              "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5],
              "max_features":["auto", "sqrt", "log2",None],
              "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90],              
              }

GV_model = GridSearchCV(DT_model, param_grid,scoring='neg_mean_squared_error',cv=5,verbose=3)

import time

s_time = time.time()
GV_model.fit(train_x, train_y)
e_time= time.time()

print("total time :", e_time-s_time)

GV_model.best_params_

tunned_DT_model = DecisionTreeRegressor(max_depth= 5,
 max_features= 'sqrt',
 max_leaf_nodes= None,
 min_samples_leaf=10,
 min_weight_fraction_leaf= 0.1,
 splitter= 'best')

tunned_DT_model.fit(train_x, train_y)
tunned_DT_Prediction = DT_model.predict(test_x)
tunned_DT_MSE = mean_squared_error(test_y, tunned_DT_Prediction)

print('MAE:', mean_absolute_error(test_y, tunned_DT_Prediction))
print('MSE:', mean_squared_error(test_y, tunned_DT_Prediction))
print('RMSE:', np.sqrt(mean_squared_error(test_y, tunned_DT_Prediction)))
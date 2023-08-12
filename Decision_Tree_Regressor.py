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
from sklearn.model_selection import cross_val_score


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
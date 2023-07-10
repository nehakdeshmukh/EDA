# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 09:57:17 2023

@author: nehak
"""

## EDA  

import pandas as pd 
import plotly.figure_factory as ff
import plotly.graph_objects as go


# read Data set

data = pd.read_csv(r"C:\Neha\kaggle Projects\Git hub\EDA\dataset/train.csv")

data.head(3)


#Given : [EC1 - EC6] are the (binary) targets, although you are only asked to predict EC1 and EC2
# Droping EC3 - EC6
data = data.drop(['id', 'EC3', 'EC4', 'EC5', 'EC6'], axis=1)


# duplicate rows 
duplicate_rows_data = data[data.duplicated()]
print("Number of duplicate rows: ", len(duplicate_rows_data))

# data types
print(data.dtypes)


# missing values
print(data.isnull().sum())

# describe data + heatmap

data_stat = data.describe()

# removed count column
data_stat = data_stat.drop("count",axis=0)


# Plto stat as a heatmap

import plotly.express as px

fig = px.imshow(data_stat.T,text_auto=True,color_continuous_scale='RdBu_r')
fig.update_layout(width=1000,height=1000)
fig.show()


# univariate Analysis

# plot bar chart 
# Get the list of column names except for "EC1"
variables = list(data.columns)


for variable in variables:
    # Check if the variable is the target column (EC1 or EC2)
    if variable == "EC1" or variable == "EC2":
        continue
    
    fig = ff.create_distplot([data[variable]],[variable])
    fig.show() 
    
    
# Box plot 

for variable in variables:
    # Check if the variable is the target column (EC1 or EC2)
    if variable == "EC1" or variable == "EC2":
        continue
    fig = go.Figure()
    fig.add_trace(go.Box(y=data[variable],name=variable))

    fig.show()
    
    
    
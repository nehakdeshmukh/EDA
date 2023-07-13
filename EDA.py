# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 09:57:17 2023

@author: nehak
"""

## EDA  

import pandas as pd 
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns


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
    
    
# count Plot for EC1

fig = go.Figure()
fig.add_trace(go.Bar(y=data[["EC1"]].value_counts(),name="EC1"))
                            
fig.show()

# count Plot for EC2

fig = go.Figure()
fig.add_trace(go.Bar(y=data[["EC2"]].value_counts(),name="EC2"))
                            
fig.show()


# combine cout plot for EC1 & EC2

fig = go.Figure()
for var in ["EC1","EC2"]:
    fig.add_trace(go.Bar(y=data[[var]].value_counts(),name=var))
                            
fig.show()


# pie Chart For EC1
fig = go.Figure(data=[go.Pie(labels=[0,1], values=data[["EC1"]].value_counts() )])
fig.show()

# pie Chart For EC2
fig = go.Figure(data=[go.Pie(labels=[0,1], values=data[["EC2"]].value_counts() )])
fig.show()

# combine Pie chart of EC1 & EC2

fig = go.Figure()
for var in ["EC1","EC2"]:
    fig.add_trace(go.Pie(labels=[0,1], values=data[[var]].value_counts(),name=var))
    fig.update_layout(title=dict(text="Pie chart of {}".format(var)))
    fig.show()
    


# Bivariate Analysis 


#violin chart for EC1

for variable in variables:
    fig = go.Figure()
    fig.add_trace(go.Violin(x=data["EC1"],y=data[variable]))
    fig.update_layout( xaxis_title="EC1",  yaxis_title=variable)
    fig.update_traces(box_visible=False, meanline_visible=True)
    fig.show()


#violin chart for EC2
for variable in variables:
    fig = go.Figure()
    fig.add_trace(go.Violin(x=data["EC2"],y=data[variable]))
    fig.update_layout( xaxis_title="EC2",  yaxis_title=variable)
    fig.update_traces(box_visible=False, meanline_visible=True)
    fig.show()



label="EC1"
# Create an empty DataFrame to store output
output_df = pd.DataFrame(columns=['Stat', '+1/-1 * ', 'Effect size', 'p-value'])

for col in data:
        if col != label:
            if data[col].isnull().sum() == 0:
                if is_numeric_dtype(data[col]):   # Calculate r and p
                    r, p = stats.pearsonr(data[label], data[col])
                    output_df.loc[col] = ['r', np.sign(r), abs(round(r, 3)), round(p,6)]
                    
output_df.sort_values(by=['Effect size', 'Stat'], ascending=[False, False])


label="EC2"
# Create an empty DataFrame to store output
output_df = pd.DataFrame(columns=['Stat', '+1/-1 * ', 'Effect size', 'p-value'])

for col in data:
        if col != label:
            if data[col].isnull().sum() == 0:
                if is_numeric_dtype(data[col]):   # Calculate r and p
                    r, p = stats.pearsonr(data[label], data[col])
                    output_df.loc[col] = ['r', np.sign(r), abs(round(r, 3)), round(p,6)]
                    
output_df.sort_values(by=['Effect size', 'Stat'], ascending=[False, False])


# correlation Heatmap


df_corr = data.corr()

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x = df_corr.columns,
        y = df_corr.index,
        z = np.array(df_corr),
        colorscale='Viridis'
    )
)



### 
# Separate the features (X) and the target variable (y) for EC1
X_ec1 = data.drop(['EC1', 'EC2'], axis=1)  # Remove 'EC1' and 'EC2' from the features
y_ec1 = data['EC1']

# Separate the features (X) and the target variable (y) for EC2
X_ec2 = data.drop(['EC1', 'EC2'], axis=1)  # Remove 'EC1' and 'EC2' from the features
y_ec2 = data['EC2']

# Create the estimator (model) for feature selection
estimator = RandomForestClassifier()  # Replace with your desired estimator

# Specify the number of features to select
num_features = 7

# Apply RFE to select the top features for EC1
rfe_ec1 = RFE(estimator, n_features_to_select=num_features)
X_rfe_ec1 = rfe_ec1.fit_transform(X_ec1, y_ec1)

# Get the mask of selected features for EC1
feature_mask_ec1 = rfe_ec1.support_

# Get the selected feature names for EC1
selected_features_ec1 = X_ec1.columns[feature_mask_ec1]

# Apply RFE to select the top features for EC2
rfe_ec2 = RFE(estimator, n_features_to_select=num_features)
X_rfe_ec2 = rfe_ec2.fit_transform(X_ec2, y_ec2)

# Get the mask of selected features for EC2
feature_mask_ec2 = rfe_ec2.support_

# Get the selected feature names for EC2
selected_features_ec2 = X_ec2.columns[feature_mask_ec2]

# Subset the dataframe with the selected features for EC1
df_selected_ec1 = data[selected_features_ec1]

# Add the EC1 column to the selected dataframe
df_selected_ec1['EC1'] = data['EC1']

# Subset the dataframe with the selected features for EC2
df_selected_ec2 = data[selected_features_ec2]

# Add the EC2 column to the selected dataframe
df_selected_ec2['EC2'] = data['EC2']

# scatter pairplot
sns.pairplot(df_selected_ec1, diag_kind='kde', hue='EC1', plot_kws={'alpha': 0.6})
plt.suptitle('Scatterplot Matrix - Top {} Features (by RFE) - EC1'.format(num_features))
plt.tight_layout()
plt.show()


# radar plot

# Create the radar plot for EC1
df_selected_ec1 = data[selected_features_ec1]
values_ec1 = df_selected_ec1.mean().values.tolist()
values_ec1 += values_ec1[:1]
features_ec1 = selected_features_ec1.tolist() + [selected_features_ec1[0]]

fig = go.Figure(data=go.Scatterpolar(
  r=values_ec1,
  theta=selected_features_ec1,
  fill='toself'
))

fig.update_layout(polar=dict(radialaxis=dict(visible=True),),showlegend=False)

fig.show()
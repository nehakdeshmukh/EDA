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

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier

from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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

# scatter pairplot EC1
sns.pairplot(df_selected_ec1, diag_kind='kde', hue='EC1', plot_kws={'alpha': 0.6})
plt.suptitle('Scatterplot Matrix - Top {} Features (by RFE) - EC1'.format(num_features))
plt.tight_layout()
plt.show()

# scatter pairplot EC2
sns.pairplot(df_selected_ec2, diag_kind='kde', hue='EC2', plot_kws={'alpha': 0.6})
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



# Create the radar plot for EC2
df_selected_ec2 = data[selected_features_ec2]
values_ec2 = df_selected_ec2.mean().values.tolist()
values_ec2 += values_ec2[:1]
features_ec2 = selected_features_ec2.tolist() + [selected_features_ec2[0]]

fig = go.Figure(data=go.Scatterpolar(
  r=values_ec2,
  theta=selected_features_ec2,
  fill='toself'
))

fig.update_layout(polar=dict(radialaxis=dict(visible=True),),showlegend=False)

fig.show()

# clustering Analysis 

def preprocess_data(data):
    # Scale the numerical features
    scaler = StandardScaler()
    numerical_features = data.columns[:-2]  
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    return data

def determine_optimal_clusters(data):
    # Determine the optimal number of clusters using the elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    
    
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, 11)), y=wcss, mode='lines'))
    fig.update_layout(xaxis_title="Number of clusters",
    yaxis_title="WCSS")
    fig.show()
    

# Preprocess the data
data = preprocess_data(data)

# Determine the optimal number of clusters
data_scaled = data.drop(['EC1', 'EC2'], axis=1)
determine_optimal_clusters(data_scaled)


# cluster plot 

fig = go.Figure()
fig.add_trace(go.Scatter(x=data["MinEStateIndex"], y=data["HallKierAlpha"],  mode='markers',marker_color=data['Cluster'],text=data['Cluster']))
fig.update_layout(xaxis_title="MinEStateIndex",yaxis_title="HallKierAlpha",title="Clusters")


fig.show()


# model Building 

# Load the training data
train_data = pd.read_csv(r"C:\Neha\kaggle Projects\Git hub\EDA\dataset/train.csv")
# Drop 'id' and 'ED' columns
train_data_E1 = train_data.drop(columns=['id','EC2', 'EC3', 'EC4', 'EC5', 'EC6'])
train_data_E2 = train_data.drop(columns=['id','EC1', 'EC3', 'EC4', 'EC5', 'EC6'])


# Preprocessing
# Let's assume all features need to be scaled
scaler = StandardScaler()
X_E1 = scaler.fit_transform(train_data_E1.drop(['EC1'], axis=1))
X_E2 = scaler.fit_transform(train_data_E2.drop(['EC2'], axis=1))


# Split the data into training and test sets for each target
X_train_EC1, X_test_EC1, y_train_EC1, y_test_EC1 = train_test_split(X_E1, train_data_E1['EC1'], test_size=0.2, random_state=42)
X_train_EC2, X_test_EC2, y_train_EC2, y_test_EC2 = train_test_split(X_E2, train_data_E2['EC2'], test_size=0.2, random_state=42)

# Define the models for EC1
model1_EC1 = GradientBoostingClassifier()

# Hyperparameter tuning with GridSearchCV
# This is just an example, you'll need to specify the parameters for your specific models
param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}

grid_search_model1_EC1 = GridSearchCV(model1_EC1, param_grid, cv=10)
grid_search_model1_EC1.fit(X_train_EC1, y_train_EC1)
print("Best parameters for model1_EC1: ", grid_search_model1_EC1.best_params_)
# Define the models for EC1 with the best parameters
model1_EC1 = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)


# Create the VotingClassifier with the best parameters
ensemble_EC1 = VotingClassifier(estimators=[('gb', model1_EC1)], voting='soft')
ensemble_EC1.fit(X_train_EC1, y_train_EC1)

# Model 2

model2_EC1 = CatBoostClassifier(verbose=False)

grid_search_model2_EC1 = GridSearchCV(model2_EC1, param_grid, cv=10)
grid_search_model2_EC1.fit(X_train_EC1, y_train_EC1)
print("Best parameters for model2_EC1: ", grid_search_model2_EC1.best_params_)

model2_EC1 = CatBoostClassifier(n_estimators=50, learning_rate=0.1, verbose=False)

# Create the VotingClassifier with the best parameters
ensemble_EC1 = VotingClassifier(estimators=[('cb', model2_EC1)], voting='soft')
ensemble_EC1.fit(X_train_EC1, y_train_EC1)


# Model 3

model3_EC1 = AdaBoostClassifier()

grid_search_model3_EC1 = GridSearchCV(model3_EC1, param_grid, cv=10)
grid_search_model3_EC1.fit(X_train_EC1, y_train_EC1)
print("Best parameters for model3_EC1: ", grid_search_model3_EC1.best_params_)

model3_EC1 = AdaBoostClassifier(n_estimators=200, learning_rate=0.1)

# Create the VotingClassifier with the best parameters
ensemble_EC1 = VotingClassifier(estimators=[ ('ab', model3_EC1)], voting='soft')
ensemble_EC1.fit(X_train_EC1, y_train_EC1)


##### Create the VotingClassifier with the best parameters for all models

# Create the VotingClassifier with the best parameters
ensemble_EC1 = VotingClassifier(estimators=[('gb', model1_EC1), ('cb', model2_EC1), ('ab', model3_EC1)], voting='soft')
ensemble_EC1.fit(X_train_EC1, y_train_EC1)


# Cross-validation
scores_EC1 = cross_val_score(ensemble_EC1, X_train_EC1, y_train_EC1, cv=10)
print("Cross-validation scores for EC1: ", scores_EC1)


# Model evaluation EC1
predictions_EC1 = ensemble_EC1.predict(X_test_EC1)
print("Classification report for EC1: ")
print(classification_report(y_test_EC1, predictions_EC1))


# Confusion matrix EC1
import seaborn as sns
cm_EC1 = confusion_matrix(y_test_EC1, predictions_EC1)
sns.heatmap(cm_EC1, annot=True, fmt='d')


# ROC curve
# This is for binary classification, you'll need to adjust for multiclass classification
probs_EC1 = ensemble_EC1.predict_proba(X_test_EC1)[:, 1]
fpr_EC1, tpr_EC1, _ = roc_curve(y_test_EC1, probs_EC1)
plt.plot(fpr_EC1, tpr_EC1, label='EC1')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Precision-recall curve
precision_EC1, recall_EC1, _ = precision_recall_curve(y_test_EC1, probs_EC1)
plt.plot(recall_EC1, precision_EC1, label='EC1')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()


# Learning curve EC1
train_sizes, train_scores, test_scores = learning_curve(ensemble_EC1, X_train_EC1, y_train_EC1, cv=10)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.show()


# Model 2

model2_EC1 = CatBoostClassifier(verbose=False)

grid_search_model2_EC1 = GridSearchCV(model2_EC1, param_grid, cv=10)
grid_search_model2_EC1.fit(X_train_EC1, y_train_EC1)
print("Best parameters for model2_EC1: ", grid_search_model2_EC1.best_params_)

model2_EC1 = CatBoostClassifier(n_estimators=50, learning_rate=0.1, verbose=False)

# Create the VotingClassifier with the best parameters
ensemble_EC1 = VotingClassifier(estimators=[('cb', model2_EC1)], voting='soft')
ensemble_EC1.fit(X_train_EC1, y_train_EC1)


# Model 3

model3_EC1 = AdaBoostClassifier()

grid_search_model3_EC1 = GridSearchCV(model3_EC1, param_grid, cv=10)
grid_search_model3_EC1.fit(X_train_EC1, y_train_EC1)
print("Best parameters for model3_EC1: ", grid_search_model3_EC1.best_params_)

model3_EC1 = AdaBoostClassifier(n_estimators=200, learning_rate=0.1)

# Create the VotingClassifier with the best parameters
ensemble_EC1 = VotingClassifier(estimators=[ ('ab', model3_EC1)], voting='soft')
ensemble_EC1.fit(X_train_EC1, y_train_EC1)


# Add Models for EC2

# Define the models for EC2
model1_EC2 = GradientBoostingClassifier()
model2_EC2 = CatBoostClassifier(verbose=False)
model3_EC2 = QuadraticDiscriminantAnalysis()


# --------------For hyperparameter grid search EC2 ---------------#

# This is just an example, you'll need to specify the parameters for your specific models
param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}

grid_search_model1_EC2 = GridSearchCV(model1_EC2, param_grid, cv=10)
grid_search_model1_EC2.fit(X_train_EC2, y_train_EC2)
print("Best parameters for model1_EC2: ", grid_search_model1_EC2.best_params_)

grid_search_model2_EC2 = GridSearchCV(model2_EC2, param_grid, cv=10)
grid_search_model2_EC2.fit(X_train_EC2, y_train_EC2)
print("Best parameters for model2_EC2: ", grid_search_model2_EC2.best_params_)

# Define the parameter grid for QuadraticDiscriminantAnalysis
param_grid_qda = {'reg_param': [0.0, 0.5, 1.0], 'tol': [0.0001, 0.001, 0.01, 0.1]}

# Perform grid search for QuadraticDiscriminantAnalysis
grid_search_model3_EC2 = GridSearchCV(model3_EC2, param_grid_qda, cv=10)
grid_search_model3_EC2.fit(X_train_EC2, y_train_EC2)
print("Best parameters for model3_EC2: ", grid_search_model3_EC2.best_params_)


# Define the models for EC2 with the best parameters
model1_EC2 = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
model2_EC2 = CatBoostClassifier(n_estimators=50, learning_rate=0.1, verbose=False)
model3_EC2 = QuadraticDiscriminantAnalysis(reg_param=0.5, tol=0.0001)

# Create the VotingClassifier with the best parameters
ensemble_EC2 = VotingClassifier(estimators=[('gb', model1_EC2), ('cb', model2_EC2), ('qda', model3_EC2)], voting='soft')
ensemble_EC2.fit(X_train_EC2, y_train_EC2)

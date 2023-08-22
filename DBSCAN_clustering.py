# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 19:20:28 2023

@author: nehak
"""

# DBSCAN clustering 
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
import time 

data_1 = pd.read_csv(r"C:\Users\nehak\Downloads\Blobs.csv")

data_2 = pd.read_csv(r"C:\Users\nehak\Downloads\Varied.csv")


fig = go.Figure()
fig.add_trace(go.Scatter(x=data_1.loc[:,'0'],y=data_1.loc[:,'1'],mode='markers'))
fig.show()
plot(fig)


fig = go.Figure()
fig.add_trace(go.Scatter(x=data_2.loc[:,'0'],y=data_2.loc[:,'1'],mode='markers'))
fig.show()
plot(fig)

eps = 5
min_samples = 10

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(data_1)

print("Cluster labels:", dbscan.labels_)

data_1.loc[:,'cluster'] = dbscan.labels_



fig = go.Figure()
fig.add_trace(go.Scatter(x=data_1.loc[:,'0'],y=data_1.loc[:,'1'],
                          mode='markers',marker_color=data_1['cluster'],
                          marker=dict(colorscale='Viridis')))
fig.show()
plot(fig)


prediction = dbscan.fit_predict(data_2)

fig = go.Figure()

fig.add_trace(go.Scatter(x=data_2.loc[:,'0'],y=data_2.loc[:,'1'],
                          mode='markers',marker_color=prediction,
                          marker=dict(colorscale='Viridis')))

fig.show()
plot(fig)

# # Plot is not showing all points belongs to same cluster 
# # Model tunning is needed 

eps_values = [0.3, 0.5, 1, 1.3, 1.5]
min_samples_values = [2, 5, 10, 20, 80]



for i in eps_values:
    for j in min_samples_values:
        dbscan = DBSCAN(eps=i, min_samples=j)
        dbscan.fit(data_2)
        prediction=dbscan.labels_
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_2.loc[:,'0'],y=data_2.loc[:,'1'],
                                 mode='markers',marker_color=prediction,
                                 marker=dict(colorscale='Viridis')))
        fig.update_layout(title ='EPS: {} & min_samples: {}'.format(i,j))
        plot(fig)
        

# from all plots I think below values are showing good clusters

dbscan = DBSCAN(eps=1.3, min_samples=10)
dbscan.fit(data_2)

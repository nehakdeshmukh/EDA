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
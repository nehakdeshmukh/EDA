
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score ,ConfusionMatrixDisplay 


data = pd.read_csv(r"C:\Users\nehak\Downloads\Iris.csv")

data_x = data.iloc[:,1:4]

data_y = data.iloc[:,-1:]


train_x,test_x, train_y, test_y = train_test_split(data_x,data_y,test_size=0.2,)



classfier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classfier.fit(train_x,train_y)

prediction = classfier.predict(test_x)

acc = accuracy_score(test_y,prediction)

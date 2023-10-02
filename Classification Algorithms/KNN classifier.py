
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score ,confusion_matrix 


data = pd.read_csv(r"C:\Users\nehak\Downloads\Iris.csv")

data_x = data.iloc[:,1:4]

data_y = data.iloc[:,-1:]


train_x,test_x, train_y, test_y = train_test_split(data_x,data_y,test_size=0.2,)


scaler = StandardScaler()

scalar_train_x = scaler.fit_transform(train_x)

scalar_test_x = scaler.fit_transform(test_x)


classfier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classfier.fit(scalar_train_x,train_y)

prediction = classfier.predict(scalar_test_x)

acc = accuracy_score(test_y,prediction)

CM = confusion_matrix(test_y,prediction)

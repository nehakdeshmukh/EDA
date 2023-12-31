import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import time
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv(r"C:\Users\nehak\Downloads\enhanced_tips_dataset.csv")

data.describe()

data.info()

data.isnull().sum()

data = data.drop(['sex','id'],axis=1)
# num_col = data.select_dtype(['int64'],'float64')

data_x = data.loc[:, data.columns != 'sex_binary']
data_y = data.loc[:, 'sex_binary']


num_col = data_x.select_dtypes(['int64', 'float64'])
cat_col = data_x.select_dtypes('object')

encoder = LabelEncoder()
cat_col = cat_col.apply(encoder.fit_transform)

data_x_updated = pd.DataFrame(pd.concat([num_col, cat_col], axis=1))

train_x, test_x, train_y, test_y = train_test_split(
    data_x_updated, data_y, test_size=0.2)


# logistic regression 

LogR_model = LogisticRegression()
LogR_model.fit(train_x,train_y)
LogR_Prediction = LogR_model.predict(test_x)
LogR_accuracy = accuracy_score(test_y,LogR_Prediction)

# Hyper parameter tunning 

logisticR_param={"solver":["newton-cg","lbfgs","liblinear","sag","saga"],
                  "penalty":["none","l1","l2","elasticnet"],
                  "C":[100,10,1.0,0.1,0.01] 
                  }
    

GV_model = GridSearchCV(LogR_model, logisticR_param,
                        cv=5, verbose=3)


s_time = time.time()
GV_model.fit(train_x, train_y)
e_time = time.time()

print("total time :", e_time-s_time)

GV_model.best_params_

TLogR_model = LogisticRegression(C=100, penalty='none', solver='sag')
TLogR_model.fit(train_x,train_y)
TLogR_Prediction = LogR_model.predict(test_x)
TLogR_accuracy = accuracy_score(test_y,LogR_Prediction)


TLogR_CM = confusion_matrix(test_y,LogR_Prediction)

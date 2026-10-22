import pandas as pd 
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.metrics import  accuracy_score
print('libraries imported  successfully')

i = load_iris()
x = i.data
y = i.target 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=2 ,random_state=42)
from sklearn.svm import SVC
svc =SVC(kernel='linear')
svc.fit(x_train,y_train)
print('accuracy score of svc', accuracy_score(y_test,svc.predict(x_test)))

import joblib as j
j.dump(svc,'svc.pkl')
print('model saved')
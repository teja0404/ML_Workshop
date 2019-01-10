import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error 

rd=pd.read_excel(r"C:\Users\TEJA\Desktop\chemf111.xlsx")
rd

x=rd.iloc[:,[4,6,7]]
x.head()
y=rd.iloc[:,[5]]
y.head()

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.4)
X_test=X_test.fillna(X_test.mean())
Y_test=Y_test.fillna(Y_test.mean())
Y_test
regr=linear_model.LinearRegression()

regr.fit(X_train,Y_train)
regr.intercept_
regr.coef_
regr.normalize
print(regr)
Y_pred=regr.predict(X_test)
print(Y_pred)
print("mean squared error:%2f"%mean_squared_error(Y_test,Y_pred))



from sklearn.ensemble import RandomForestClassifier
rm=RandomForestClassifier(n_estimators=100)
rm.fit(X_test,Y_test)
predicted=rm.predict(X_train)
predicted
print("mean squared error:%2f"%mean_squared_error(Y_test,predicted))

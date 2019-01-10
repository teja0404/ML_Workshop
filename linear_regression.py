# -*- coding: utf-8 -*-
"""
Created on Wed May 23 19:17:56 2018

@author: TEJA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error 
bl=pd.read_excel(r"C:/Users/TEJA/Desktop/PHYF111.xlsx")

bl.head()
bl.drop()
bl.shape
X=bl.iloc[:,(6)]
X
Y=bl.ix[:,(10)]
Y.head(3)
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)
regr=linear_model.LinearRegression()
regr.fit(X_train,Y_train)
regr.intercept_
print(regr)
Y_pred=regr.predict(X_test)
print(Y_pred)
print("mean squared error:%2f"%mean_squared_error(Y_test,Y_pred))

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:11:15 2018

@author: TEJA
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
import lightgbm as lgb
dff=pd.read_csv(r"C:\Users\TEJA\Desktop\train.csv")
dff.info()
print(dff[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean())
dff[["Pclass","Survived"]]
dff["Age"]=dff["Age"].fillna(np.mean(dff["Age"]))
dff["Age"].describe()
np.median(dff["Age"])


imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(dff[["Age"]])
dff["Age"]=imputer.fit_transform(dff[["Age"]])
dff
X=dff.ix[:,[2,4,5,6,7,9]]
Y=dff.ix[:,[1]]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X["Sex"] = labelencoder_X_1.fit_transform(X["Sex"])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=80)

model=RandomForestClassifier(n_estimators=10,random_state=0)
model.fit(X_train,Y_train)
Y_predict=model.predict(X_test)
cm=confusion_matrix(Y_predict,Y_test)
cm

model2 = XGBClassifier()
model2.fit(X_train,Y_train)
Y2_predict=model2.predict(X_test)
cm2=confusion_matrix(Y_predict,Y_test)
cm2

























pred=pd.read_csv(r"C:\Users\TEJA\Desktop\test.csv")
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(pred[["Age"]])
pred["Age"]=imputer.fit_transform(pred[["Age"]])
imputer=imputer.fit(pred[["Fare"]])
pred["Fare"]=imputer.fit_transform(pred[["Fare"]])
X2=pred.iloc[:,[1,3,4,5,6,8]]
X2["Sex"] = labelencoder_X_1.fit_transform(X2["Sex"])
onehotencoder = OneHotEncoder(categorical_features = [1])
X2 = onehotencoder.fit_transform(X2).toarray()
Y2_predict=model.predict(X2)
np.shape(Y2_predict)ss
























data={'row_id':pred["PassengerId"],'survived':Y2_predict}
sub = pd.DataFrame(data )
sub.to_csv(r'C:\Users\TEJA\Desktop\Untitled spreadsheet - Sheet1.csv', index=False)




cm=confusion_matrix(Y_predict,Y_test)
cm




imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(dff[["Age"]])
dff["Age"]=imputer.fit_transform(dff[["Age"]])
dff["Embarked"]=dff.ix[~dff["Embarked"].isnull()]
X=dff.iloc[:,[2,5,6,7,9]]
np.shape(dff)
np.shape(X)
Y=dff.iloc[:,1]
Y
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
labelencoder = LabelEncoder()
sc = StandardScaler()

X = sc.fit_transform(X)

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=80)
model=RandomForestClassifier(n_estimators=10,random_state=0)
model.fit(X_train,Y_train)
Y_predict=model.predict(X_test)
Y_predict
cm=confusion_matrix(Y_predict,Y_test)
cm
pred=pd.read_csv(r"C:\Users\TEJA\Desktop\test.csv")
pred
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(pred[["Age"]])
pred["Age"]=imputer.fit_transform(pred[["Age"]])
imputer=imputer.fit(pred[["Fare"]])
pred["Fare"]=imputer.fit_transform(pred[["Fare"]])
X2=pred.iloc[:,[1,4,5,6,8]]
X2.info()
X2= sc.fit_transform(X2.astype(str))
X2
Y2_predict=model.predict(X2)
np.shape(Y2_predict)
Y2_predict
data={'row_id':pred["PassengerId"],'survived':Y2_predict}
sub = pd.DataFrame(data )
sub.to_csv(r'C:\Users\TEJA\Desktop\Untitled spreadsheet - Sheet1.csv', index=False)





from xgboost import XGBClassifier
model2 = XGBClassifier()
model2.fit(X_train,Y_train)
model2.predict(X_test)
cm2=confusion_matrix(Y_predict,Y_test)
cm2

z=dff.iloc[:,[2,4]]
labelencoder_X_1 = LabelEncoder()
z.values[:,0] = labelencoder_X_1.fit_transform(z.values[:,0])#visual
labelencoder_X_2 = LabelEncoder()
z.values[:, 1] = labelencoder_X_2.fit_transform(z.values[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
z = onehotencoder.fit_transform(z.astype(str)).toarray()
z
z = z[:, 1:]

import scipy
import scipy.stats
df=pd.read_csv(r"C:\Users\TEJA\Desktop\Loan_Train.csv")
z=df.iloc[:,[1,2]]
z
labelencoder_X_1 = LabelEncoder()
z["Gender"] = labelencoder_X_1.fit_transform(z["Gender"].astype(str))
z["Gender"]
z["Married"] = labelencoder_X_1.fit_transform(z["Married"].astype(str))
z["Married"]
z
z = onehotencoder.fit_transform(z.astype(str)).toarray()
z=pd.DataFrame(z)
z["Married"]
df=df.ix[~df["LoanAmount"].isnull()]
df.fillna(0)
df
pairedsam = scipy.stats.ttest_rel(df["ApplicantIncome"],df["Self_Employed"].astype(str))
pairedsam


from xgboost.sklearn import XGBClassifier
xg=XGBClassifier()
xg.fit
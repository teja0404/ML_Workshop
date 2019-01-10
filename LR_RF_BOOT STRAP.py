# -*- coding: utf-8 -*-
"""
Created on Sun May 27 10:10:55 2018

@author: TEJA
"""

#logistic regression
import pandas as pd
import numpy as np
import seaborn as sb
import sklearn
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report,auc,roc_curve
import pylab as pl
import statsmodels.api as sm
bl=pd.read_csv(r"C:\Users\TEJA\Desktop\bankloan.csv")
bl=pd.DataFrame(bl)
bl.shape
bl.info
plt.plot(bl.default,'bo')
bl_data=bl.drop(['address','ed','debtinc','employ'],1)
bl_data.head(3)
x=bl_data.ix[:,(0,1,2,3)]
y=bl_data.ix[:,4]
y.head(3)
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.4)
len(X_train)
len(Y_test)
clf=LogisticRegression(fit_intercept=True,C=1e15)
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
y_pred
y_pred.sum()
from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(Y_test,y_pred)
confusion_matrix
print(classification_report(Y_test,y_pred))
preds1=clf.predict_proba(X_test)[:,1]
fpr1,tpr1,thresholds1=metrics.roc_curve(Y_test,y_pred)
df1=pd.DataFrame(dict(fpr=fpr1,tpr=tpr1))
aucl=auc(fpr1,tpr1)
plt.figure()
lw=2
plt.plot(fpr1,tpr1,color='red',lw=lw,label='ROC curve(area=%0.2f)'%aucl)
plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.5])
plt.xlabel(['false'])
plt.ylabel(['true'])
plt.show()


#chi squared distribution
#test works for non parametric data where we donot follow any assump about normality and population parameters is called chi suared test
import pandas as pd
import numpy as np
import scipy
cs2m=pd.read_csv(r"C:\Users\TEJA\Desktop\cs2M.csv")
pd.crosstab(cs2m.AnxtyLH,cs2m.DrugR,margins=True)
AnxtyDrug=np.array([[11,5],[4,10]])
scipy.stats.chi2_contingency(AnxtyDrug)



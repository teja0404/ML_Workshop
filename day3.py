# -*- coding: utf-8 -*-
"""
Created on Fri May 25 14:09:21 2018

@author: TEJA
"""

import scipy.stats
import pandas as pd
import numpy as np
df=pd.read_excel(r"C:\Users\TEJA\Desktop\cs2M.xls")
onesam=scipy.stats.ttest_1samp(a = df.Age,popmean=40)
print(onesam)



#for comparing
pd=scipy.stats.ttest_rel(df.BP,df.Chlstrl)
print(pd)

#for same person 
df.shape
df_x= df[df.AnxtyLH==0]#seperating with specific values
df_x
df_y=df[df.AnxtyLH==1]
df_y.shape
scipy.stats.ttest_ind(df_x.BP,df_y.BP)

#one way anova
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod=ols('AnxtyLH~DrugR',data=df).fit()
aov_table = sm.stats.anova_lm(mod,type = 2)
print(aov_table)
import pandas as pd
rb=pd.read_csv(r"C:\Users\TEJA\Downloads\Student.csv")


#testing the stats 
import scipy.stats
import pandas as pd
import numpy as np
df["LoanAmount"]=df.ix[df["LoanAmount"].isnull()==0]    
df["LoanAmount"]
df=pd.read_csv(r"C:\Users\TEJA\Desktop\train.csv")
df2=pd.read_csv(r"C:\Users\TEJA\Desktop\test.csv")
df.info()
df2.info()
np.shape(df)


df["CoapplicantIncome"]
pred=scipy.stats.ttest_rel(df["LoanAmount"].astype(float),df["CoapplicantIncome"].astype(str))
pred
df.info()
np.shape(df)
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(df[["Age"]])
df["Age"]=imputer.fit_transform(df[["Age"]])
df["Age"]

from xgboost import XGBClassifier
model2 = XGBClassifier()
model2.fit(X_train,Y_train)
model2.predict()
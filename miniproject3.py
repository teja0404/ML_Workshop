
from scipy import stats
import pandas as pd
import numpy as np
grades=pd.read_excel("C:/Users/TEJA/Desktop/grades.xls")

grades.head()
pairedsam=stats.ttest_rel(grades.quiz1,grades.quiz2)

print(pairedsam)

#as pvalue is less than 0.05 we reject the null hypothesis and accept the alternate hypothesis

import statsmodels.api as sm

from statsmodels.formula.api import ols
grades.head()
mod=ols('gender~grade',data=grades).fit()
print(mod)

pd.crosstab(grades.lowup,grades.gender,margins=True)

array=np.array([[16,6],[48,35]])
array

stats.chi2_contingency(array)

X=grades[["final"]]

y=grades[["ethnicity"]]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4) 

# Call the decision tree
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)

# Predicting the variable
treepred = dtree.predict(X_test)
treepred
y_test
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, treepred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, treepred)


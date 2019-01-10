# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:44:04 2018

@author: TEJA
"""

BITS - Machine Learning Batch 2
BITS - Machine Learning
Batch 2
STREAM
CLASSMATES
ABOUT
BITS - Machine Learning
Batch 2

Parvez Nadvi

Lokesh Payasi
UPCOMING
Woohoo, no work due soon!
VIEW ALL
TOPICS
Exam

Post by Kaluvala Dheeraj
Kaluvala Dheeraj
Created 8:19 PM8:19 PM
Sir, Can you keep the Notes to prepare for Exam Tomorrow

Announcement: "We will be having a group photo of all…"
Parvez Nadvi
Created 4:00 PM4:00 PM
We will be having a group photo of all the courses ..kindly stay back and assemble in the open air theatre.
Thank you.

Announcement: "If anyone's name isn't present in the…"
Parvez Nadvi
Created 3:29 PM3:29 PM
Exam
If anyone's name isn't present in the batch please come to PRANAV and give in your details. ASAP

Yaswanth Reddy3:43 PM
Lilith Avinash

Announcement: "Exam Batch Details. Please report to…"
Parvez Nadvi
Created 3:21 PM3:21 PM
Exam
Exam Batch Details.
Please report to the respected allocated rooms for your batch 15 mins prior.
Please go through the exam schedule and note down your batch number and timings and room number. 
If there is any doubt please come and ask me immediately. Late queries will not be entertained.

HYD ML.xlsx
Excel

Anand Sangabattuni4:41 PM
Do we have to stay back after the exam or can we leave??

Announcement: "Text file for python"
Lokesh Payasi
Created 8:15 AM8:15 AM
Text file for python

metamorphosis_clean.txt
Text

Announcement: "dataset for text analytics"
Lokesh Payasi
Created 7:46 AM7:46 AM
dataset for text analytics

sms_spam.csv
Comma Separated Values

Announcement: "for todays class install tm, SnowballC,…"
Lokesh Payasi
Created 7:36 AM7:36 AM
for todays class install tm, SnowballC, wordcloud, NLP package in r

Announcement: "PPTs and Datasets-2"
Lokesh Payasi
Created May 28May 28
PPTs and Datasets-2

Decision Trees_Lok.pptx
PowerPoint

Machine Learning.pptx
PowerPoint

Support Vector Machines_Steve.pptx
PowerPoint

boosting.doc
Word

Machine learning Course Outline.docx
Word

packages.docx
Word

grad_desc.R
Binary File

knn_predict.R
Binary File

log_reg.R
Binary File

nn_mtcars.R
Binary File

rpart_bankloan.R
Binary File

bankloan coding caret and train.R
Binary File

bankloan coding caret and train(1).R
Binary File

decisiontree.R
Binary File

DecisionTree_280717.pptx
PowerPoint

student.csv
Comma Separated Values

Log_Reg1.R
Binary File

Logistic Regression.pptx
PowerPoint

Logistic Regression_Updated.pptx
PowerPoint

Announcement: "PPTs and Datasets"
Lokesh Payasi
Created May 28May 28
PPTs and Datasets

important packages for R and python.docx
Word

Linear Regression.pptx
PowerPoint

Bagging.pptx
PowerPoint

Ensemble in Python.ppt
PowerPoint

Python Basics.pptx
PowerPoint

Python Numpy.pptx
PowerPoint

SVM Python.pptx
PowerPoint

Decision Tree Python.pptx
PowerPoint

DT with Python.py
Text

Hypothesis Testing in Python.pptx
PowerPoint

KYD Python.pptx
PowerPoint

LogRegression by Python.pptx
PowerPoint

Announcement: "PPTs and Datasets"
Lokesh Payasi
Created May 28May 28
PPTs and Datasets

Assignment: "2nd Min project"
Lokesh Payasi
Created May 27May 27
DONE
2nd Min project
do it in R and python, check out project 2

Mini Projects.ppt
PowerPoint

Announcement: "list of packages for machine learning…"
Lokesh Payasi
Created May 27May 27
list of packages for machine learning in R and Python

important packages for R and python.docx
Word

Announcement: "new dataset for knn"
Lokesh Payasi
Created May 26May 26
new dataset for knn

ppt Knn.pptx
PowerPoint

wisc_bc_data.csv
Comma Separated Values

Announcement: "new dataset for knn"
Lokesh Payasi
Created May 26May 26
new dataset for knn

Announcement: "PPTs till 25th MAy"
Lokesh Payasi
Created May 26May 26
PPTs till 25th MAy

Linear Regression.pptx
PowerPoint

Python Basics.pptx
PowerPoint

Python Numpy.pptx
PowerPoint

Hypothesis Testing in Python.pptx
PowerPoint

UNIT-1.ppt
PowerPoint

UNIT-5.pptx
PowerPoint

Descriptive Statistics.pptx
PowerPoint

Announcement: "Min-Project-1"
Lokesh Payasi
Created May 26May 26
Min-Project-1

Mini Projects.ppt
PowerPoint

Announcement: "Data-set for Assignment-1(Not Project)-…"
Lokesh Payasi
Created May 25May 25
Data-set for Assignment-1(Not Project)- you can select any two file for doing your assignment. for Decision Tree Diagram you should select Assignment.

Assignment.csv
Comma Separated Values

bankloan.csv
Comma Separated Values

creditset.csv
Comma Separated Values

Announcement: "The Data-set for all the classes"
Lokesh Payasi
Created May 25May 25
The Data-set for all the classes

excelTOspss.xlsx
Excel

cs2M.xls
Excel

grades.xls
Excel

bankloan.xls
Excel

oil_temp.xls
Excel

project_kochu.xls
Excel

coal.csv
Comma Separated Values

flight_accident.csv
Comma Separated Values

gapC.csv
Comma Separated Values

Nile.csv
Comma Separated Values

Oil.csv
Comma Separated Values

pension.csv
Comma Separated Values

Project_Ann.csv
Comma Separated Values

project_kochu.csv
Comma Separated Values

Student.csv
Comma Separated Values

youth_survey.csv
Comma Separated Values

zip_code.csv
Comma Separated Values

tecator.csv
Comma Separated Values

spambase.csv
Comma Separated Values

# -*- coding: utf-8 -*-
"""
Created on Thu May 24 06:56:51 2018

@author: Lucky
"""

import sklearn.datasets as datasets
import pandas as pd
iris=datasets.load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
y=iris.target


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y,test_size = .4)

len(X_train)
len(y_test)
len(X_test)
X_test.head()
y_test.head()

# Call the decision tree
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)

# Predicting the variable
treepred = dtree.predict(X_test)
treepred
treepred.sum()
y_test.sum()

# confusion Matrix

cm_deci1 = pd.crosstab(y_test, treepred, rownames=['True'],colnames=['Predicted'], margins=True)
cm_deci1

# Graph
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data,filled=True, rounded=True,special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())





dtree
# converting into the pdf file
with open("dtree.dot", "w") as t:
    t = tree.export_graphviz(dtree, out_file=t)

dot -Tpdf dtree.dot -o dtree.pdf
open -a preview dtree.pdf
DT with Python.py
Displaying DT with Python.py.
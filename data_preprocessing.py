import pandas as pd
import numpy as np
train=pd.read_csv(r"C:\Users\TEJA\Desktop\train (1).csv")
train.info()
train.describe()
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
test=pd.read_csv(r"C:\Users\TEJA\Desktop\test (1).csv")
test.info()
train.columns[train.isnull().any()]
miss=train.isnull().sum()
miss=miss/len(train)
miss=miss[miss>0]
sns.distplot(train["SalePrice"])
target=np.log(train["SalePrice"])
numeric_data=train.select_dtypes(include=[np.number])
cat_data=train.select_dtypes(exclude=[np.number])
del numeric_data["Id"]
corr=numeric_data.corr()
sns.heatmap(corr)
print(corr["SalePrice"].sort_values(ascending=False))
stats.mode

train["LotShape"] 
train["LandSlope"] 
train["2ndFlrSF"]

#functions in python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
def factorize(data,var,fill_na=None):
    if fill_na is not None:
        data[var].fillna(fill_na,inplace=True)
    le.fit(data[var])
    data[var]=le.transform(data[var])
    return data
train['LotFrontage']
train['Neighborhood']

name = np.array(['ExterQual','PoolQC' ,'ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu', 'GarageQual','GarageCond'])
name
qual_dict = {np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
for i in name:
     train[i] = train[i].map(qual_dict).astype(int)
train["ExterCond"] 
train["BsmtFinType1"]
train["MSZoning"]
train["LotShape"]
train["MasVnrArea"]
train["WoodDeckSF"] 



def onehot(onehot_df, df, column_name, fill_na):
       onehot_df[column_name] = df[column_name]
       if fill_na is not None:
            onehot_df[column_name].fillna(fill_na, inplace=True)

       dummies = pd.get_dummies(onehot_df[column_name], prefix="_"+column_name)
       onehot_df = onehot_df.join(dummies)
       onehot_df = onehot_df.drop([column_name], axis=1)
       return onehot_df
 onehot_df = onehot(onehot_df, df, "MSSubClass", None)

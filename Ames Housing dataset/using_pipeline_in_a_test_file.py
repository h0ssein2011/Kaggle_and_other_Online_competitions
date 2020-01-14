import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
# 



train =pd.read_csv('Input/train.csv')
test=pd.read_csv('Input/test.csv')
# print(train.columns)
X = train.drop('SalePrice',axis = 1)

y=train['SalePrice']

X_train ,X_valid,y_train,y_valid = train_test_split(X,y,random_state=0 ,train_size =0.7 )

#select cols with null vals
null_cols = [col for col in X_train.columns if X_train[col].isnull().any()>0]

#select low cardiality cols 
cat_cols = [cname for cname in X_train.columns if X_train[cname].dtype == 'object' and X_train[cname].nunique() < 10 ]
int_cols = [col for col in X_train.columns if X_train[col].dtype in ('int64','float64')]

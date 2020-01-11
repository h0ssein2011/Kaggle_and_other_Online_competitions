
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df=pd.read_csv('Input/melb_data.csv')
print(df.columns)
#
X=df.drop('Price',axis=1)
y=df['Price']
#
# #
X_train,X_valid,y_train,y_valid =train_test_split(X,y,random_state=0,train_size=0.8)
#
# return cols with null vals
null_cols = [col for col in X_train.columns if X_train[col].isnull().any()]
print('null cols are:',null_cols)


X_train_reduced = X_train.drop(null_cols ,axis=1, inplace =True)
X_valid_reduced = X_valid.drop(null_cols ,axis=1, inplace =True)

#select low cardinality cols
low_card_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and X_train[cname].dtype == 'object']
int_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ('int64','float64')]
# keep selected columns only
my_cols = low_card_cols + int_cols

X_train = X_train[my_cols]
X_valid = X_valid[my_cols]

# list of categorical vars
s =(X_train.dtypes == 'object')
t=list(s[s].index)
# object_cols =list(s[s].index)
print(t)
# print('object cols are:'.format(object_cols))

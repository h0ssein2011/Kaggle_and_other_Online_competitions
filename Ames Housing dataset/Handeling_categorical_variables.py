
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
# s=(X_train.dtypes == 'object')
# object_cols = list(s[s].index)
#
# print('object_cols are:{}'.format(object_cols))



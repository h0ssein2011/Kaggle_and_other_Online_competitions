
import pandas as pd
import numpy as np


df=pd.read_csv('Input/train.csv' , index_col='Id')
df_test = pd.read_csv('Input/test.csv' , index_col='Id')

X=df.drop('SalePrice',axis=1)
y=df['SalePrice']


#

s=(X.select_dtypes == 'object')
object_cols = list(s[s].index)

print('object_cols are:{}'.format(object_cols))



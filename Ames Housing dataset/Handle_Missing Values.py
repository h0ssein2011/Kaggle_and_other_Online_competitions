import  pandas as pd
from sklearn.ensemble import  RandomForestRegressor

df=pd.read_csv('Input/train.csv' , index_col='Id')
df_test = pd.read_csv('Input/test.csv' , index_col='Id')
print(df.isnull().sum())

X=df.drop('SalePrice',axis=1)
y=df['SalePrice']


#for simplicity we just keep numerical vals
X=X.select_dtypes(exclude='object')

#do above for test dataset
X_test=df_test.drop('SalePrice',axis=1)
y_test=df_test['SalePrice']


#for simplicity we just keep numerical vals
X=X_test.select_dtypes(exclude='object')





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
object_cols=list(s[s].index)


#import Model and Metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

#approach 1:delete categorical variables
droped_X_train = X_train.select_dtypes(exclude = 'object')
droped_X_valid = X_valid.select_dtypes(exclude = 'object')

drop_MAE = score_dataset(droped_X_train ,droped_X_valid,y_train,y_valid)
print('MAE dropped approach: {}'.format(drop_MAE))


#approach2:label encoding
label_X_train = X_train.copy()
label_X_valid =X_valid.copy()
from sklearn.preprocessing import LabelEncoder

Encoder =LabelEncoder()

for col in object_cols:
    label_X_train[col]=Encoder.fit_transform(label_X_train[col])
    label_X_valid[col]=Encoder.fit_transform(label_X_valid[col])

label_approach_MAE= score_dataset(label_X_train,label_X_valid,y_train,y_valid)
print('MAE lable endcoding approach: {}'.format(label_approach_MAE))



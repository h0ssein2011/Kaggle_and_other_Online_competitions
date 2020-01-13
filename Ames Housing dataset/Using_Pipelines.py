
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


#using ColumnTransformer to bundle multi preprocessing and model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

#preprocessing for numerical data
Simple_imputer = SimpleImputer(strategy='constant')

#preprocessing for categorical variables

Categorical_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                                          ('onehot',OneHotEncoder(handle_unknown='ignore'))])


#Bundle preprocessing for numerical and categorical variables
preprocesser = ColumnTransformer(
    transformers= [
        ('num',Simple_imputer ,int_cols),
        ('cat',Categorical_transformer ,low_card_cols)
    ]
)


#define the model
from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor(n_estimators=100 , random_state=0)

#evaluate the model in the pipline
my_pipeline = Pipeline(steps=[('preprocessor',preprocesser) ,('model',RF_model)])

my_pipeline.fit(X_train,y_train)
pred=my_pipeline.predict(X_valid)

# calculate MAE
from sklearn.metrics import mean_absolute_error
score = mean_absolute_error(y_valid,pred)

print('MAE :',score)








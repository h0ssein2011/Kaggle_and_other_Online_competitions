#0- load db
import pandas as pd 
import numpy as np 

#1- select X & y
df = pd.read_csv('Input/melb_data.csv')

#select some cols
selected_cols = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']

X=df[selected_cols]
y=df['Price']


from sklearn.model_selection import  train_test_split
X_train,X_valid ,y_train,y_valid =train_test_split(X,y,train_size=0.7 , random_state=0)
#2- find null values

#3- find type vars (int/cats)
#4- preprocessing for each type
#5- build/fit model

from xgboost import XGBRegressor

my_model = XGBRegressor()

my_model.fit(X_train,y_train)
preds = my_model.predict(X_valid)

# 6- ealuation 

from sklearn.metrics import mean_absolute_error
MAE= mean_absolute_error(y_valid, preds)
print('MAE xgboost is:{}'.format(MAE))

#compare with RF 
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
X_valid = pd.DataFrame(my_imputer.fit_transform(X_valid))



RF_model =RandomForestRegressor()
RF_model.fit(X_train,y_train)
RF_preds = RF_model.predict(X_valid)
RF_MAE = mean_absolute_error(y_valid, RF_preds)
print('MAE RF is: {}'.format(RF_MAE))

#use Xgboost with Parameters

modified_model= XGBRegressor(n_estimators=1000 , learning_rate= 0.05,n_jobs=4)
modified_model.fit(X_train,y_train,early_stopping_rounds=5,
                    eval_set=[(X_valid,y_valid)],
                    verbose=False)

modified_preds = modified_model.predict(X_valid)
MAE_Xgb_modified =mean_absolute_error(y_valid,modified_preds)
print('MAE Xgb_modified is:{}'.format(MAE_Xgb_modified))

from xgboost import xg
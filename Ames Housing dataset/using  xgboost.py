#0- load db
import pandas as pd 
import numpy as np 

#1- select X & y
df = pd.read_csv('Input/melb_data.csv')

#select some cols
selected_cols = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']

X=df[selected_cols]
y=df.Price 

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

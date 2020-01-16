

import pandas as pd 
import numpy as np 



#0- load db
df= pd.read_csv('./Input/melb_data.csv')
#1- select X & y
cols_to_use=['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X= df[cols_to_use]
y=df['Price']

#1- find null values
null_cols = [col for col in X if X[col].isull().any()]
#1.1- fill nulls 

# postpone to pipeline step

#2- find type vars
#2-1 cat vars and low cardinlity
# not needed here as all data are ints


#3- preprocessing 
#3.1 create pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessing',SimpleImputer),
                                ('model',RandomForestRegressor(n_estimators=50 , random_state=0))])

#3.2 Corss validation
from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(my_pipeline , X,y, cv=5 ,scoring='')
#4- Fit model / 


# 5- ealuation 


#6 submuission (if so)


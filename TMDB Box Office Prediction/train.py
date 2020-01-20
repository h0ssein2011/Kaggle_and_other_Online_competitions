

#0- load db
import pandas as pd 
import numpy as np 

train_data = pd.read_csv('Input/train.csv',index_col='id')
test_data = pd.read_csv('Input/test.csv',index_col='id')

# 0.1 data analysis 
train_data['collection_name']=np.nan
train_data['genre_name']=np.nan
train_data['Production_company_name']=np.nan
train_data['has_home_page']=np.where(train_data.homepage.isnull(),0,1)
train_data['Spoken_language_name']=np.nan
train_data['has_tag_line']=np.where(train_data.tagline.isnull(),0,1)
# 

#1- select X & y


    #1.1- fill nulls 
#TBD
#2- find null values
#TBD

#3- find type vars (int/cats)
#select numerical colls only for simplicity
# num_cols = [col for col in train_data.columns if train_data[col].dtype in ['int64','float64']]
num_cols = ['budget', 'popularity', 'runtime', 'has_home_page', 'Spoken_language_name','has_tag_line']
X=train_data[num_cols]

y=train_data.revenue

from sklearn.model_selection import train_test_split
X_train, X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.33, random_state=42)


#4- preprocessing for each type
#5- build a pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


my_pipeline = Pipeline(steps=[('preprocessor',SimpleImputer()),('model',RandomForestRegressor(n_estimators=50 ,random_state=42))])
my_pipeline.fit(X_train,y_train)
preds = my_pipeline.predict(X_valid)
# 6- ealuation 

from sklearn.metrics import mean_absolute_error

MAE = mean_absolute_error(y_valid,preds)
print('MAE is:',MAE)
#7 submuission (if so)
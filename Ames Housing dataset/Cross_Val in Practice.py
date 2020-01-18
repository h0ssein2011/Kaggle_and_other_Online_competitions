import pandas as pd 
import numpy as np

#0- load db
train =pd.read_csv('./Input/train.csv',index_col='Id')
test =pd.read_csv('./Input/test.csv',index_col='Id')

#1- select X & y
X_train =train.drop('SalePrice',axis=1)
y=train['SalePrice']


#2- find null values

    #2.1- fill nulls 

#3- find type vars (int/cats)
#here just dind num variables
num_cols =  [col for col in X_train.columns if X_train[col].dtype in (['int64','float64'])]
#select num_cols

X_train=X_train[num_cols].copy()
X_test=test[num_cols].copy()
#4- preprocessing for each type
# build a pipeline
#5- build model

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

my_pipeline =Pipeline(steps=[('preprocessor',SimpleImputer()),('model',RandomForestRegressor(n_estimators=50 , random_state= 0))])

# 6- ealuation 
## use cross-validation 

from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(my_pipeline ,X_train,y,cv=5 ,scoring='neg_mean_absolute_error')
print('MAE is:',scores.mean())

#7 using CV to find the best n_estimators

#create a function to get MAE scores for the given n_estimators

def get_score(n_estimators):

    my_pipeline = Pipeline(steps=[('preprocesor',SimpleImputer())
                                    ,('model',RandomForestRegressor(n_estimators=n_estimators,
                                                                                            random_state=0))])

    scores = -1 * cross_val_score(my_pipeline, X_train,y, cv=3,scoring='neg_mean_absolute_error')

    return(scores.mean())


# store result scores in a dictionary for 50 to 400 by steps 50
results = {}
for i in range(1,9):
    results[i * 50]= get_score(i*50)


#shows the results in a graph
import matplotlib.pyplot as plt 

plt.plot(list(results.keys()), list(results.values()))
plt.show()

#it seems that the best n for RandomForest is 200

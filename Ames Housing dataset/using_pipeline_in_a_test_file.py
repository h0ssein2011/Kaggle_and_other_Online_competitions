import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
# 

train =pd.read_csv('Input/train.csv')
test=pd.read_csv('Input/test.csv',index_col='Id')
# print(train.columns)
X = train.drop('SalePrice',axis = 1)

y=train['SalePrice']

X_train ,X_valid,y_train,y_valid = train_test_split(X,y,random_state=0 ,train_size =0.7 )

#select cols with null vals
null_cols = [col for col in X_train.columns if X_train[col].isnull().any()>0]

X_train.drop(null_cols ,axis = 1 , inplace =True)
X_valid.drop(null_cols ,axis = 1 , inplace =True)
    
#select low cardiality cols 
cat_cols = [cname for cname in X_train.columns if X_train[cname].dtype == 'object' and X_train[cname].nunique() < 10 ]
int_cols = [col for col in X_train.columns if X_train[col].dtype in ('int64','float64')]

my_cols = cat_cols + int_cols

#replace cat and int cols with current cols
X_train=X_train[my_cols]
X_valid = X_valid[my_cols]


## preprocessing part
# import required packages
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

#import pipeline packaegs
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#int cols preprpcessing
num_transfomer = SimpleImputer(strategy='constant')

#catecorical values
cat_preprocessings = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                                     ('one_hot',OneHotEncoder(handle_unknown='ignore'))])


#bundle all preprocessing in a single line!
preprcessing_part = ColumnTransformer(transformers=[('num',num_transfomer,int_cols),('cat',cat_preprocessings,cat_cols)])

#define the model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100 , random_state=0)

#bundel prepocessing and model in one line 
clf =Pipeline(steps=[('preprocess',preprcessing_part),('model',model)])
clf.fit(X_train,y_train)

#predict by pipelined model
pereds = clf.predict(X_valid)

#evaluate the model
from sklearn.metrics import mean_absolute_error

print('MAE Pipelined model is',mean_absolute_error(y_valid,pereds))

perd_test = clf.predict(test)
df_output =pd.DataFrame(perd_test,columns=['SalePrice'])
df_output['Id']=test.index
df_output =df_output[['Id','SalePrice']]
df_output.to_csv('Input/PipelineSubmision.csv',index=False)

print('submission file is in Input directory')
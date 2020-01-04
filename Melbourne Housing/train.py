
# https://www.kaggle.com/alexisbcook/missing-values

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('melb_data.csv')

# Select target
y = data.Price

# To keep things simple, use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

# fucntion to measure the models performance

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# def score_dataset(X_train, X_valid,y_train, y_valid):
#     model = RandomForestRegressor(n_estimators=10 ,random_state=0 )
#     model.fit(X_train , y_train)
#     preds = model.predict(X_valid)
#     return mean_absolute_error(y_valid, preds)


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
#approaches with null vlas columns

#1-delete cols with null vals


cols_with_missing_vals = [col for col in X_train.columns if X_train[col].isnull().any()]

reduced_X_train =X_train.drop(cols_with_missing_vals , axis=1)
reduced_X_valid =X_valid.drop(cols_with_missing_vals , axis=1)

reduced_error = score_dataset(reduced_X_train ,reduced_X_valid,y_train,y_valid)
#
print('MAE in dropping cols with null values :{}'.format(reduced_error))

# 2-Imputation

from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()

Imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
Imputed_X_valid = pd.DataFrame(my_imputer.fit_transform(X_valid))


#back col names
Imputed_X_train.columns = X_train.columns
Imputed_X_valid.columns = X_valid.columns

MAE_imputation=score_dataset(Imputed_X_train , Imputed_X_valid ,y_train , y_valid )

print('MAE for imputation approach is: {}'.format(MAE_imputation))
print('Imputation has better performance')


#approach 3: extention of imputation
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

for col in cols_with_missing_vals:
    X_train_plus[col + '_was missing'] =X_train_plus[col].isnull()
    X_valid_plus[col + '_was missing'] =X_valid_plus[col].isnull()

my_imputer = SimpleImputer()

imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
# imputed_X_valid_plus = pd.DataFrame(my_imputer.fit_transform(X_valid_plus))


rename the columns
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_train_plus.columns = X_train_plus.columns

MAE_imputation_plus=score_dataset(imputed_X_train_plus,imputed_X_valid_plus,y_train,y_valid)
print('MAE approach 3(extension of imputation is :{}'.format(MAE_imputation_plus))








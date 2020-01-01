import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('Melbourne_housing_extra_data.csv')

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

def score_dataset(X_train, X_valid,y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10 ,random_state=0 )
    model.fit(X_train , y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

#approaches with null vlas columns

#1-delete cols with null vals


cols_with_missing_vals = [col for col in X_train.columns if X_train[col].isnull().any()]

reduced_X_train=X_train.drop(cols_with_missing_vals , axis=1)
reduced_X_valid=X_valid.drop(cols_with_missing_vals , axis=1)

reduced_error = score_dataset(reduced_X_train ,reduced_X_valid,y_train,y_valid)
#
# print('Meae in dropping cols with null values :{}'.format(reduced_error))


# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

#2-Imputation



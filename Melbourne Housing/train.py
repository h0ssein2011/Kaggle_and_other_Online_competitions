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

from sklearn.ensmeble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mea

def score_datasets(X_train, X_valid,y_train, y_valid ):
    model = RandomForestRegressor(n_estim)
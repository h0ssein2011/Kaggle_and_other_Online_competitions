import  pandas as pd

df=pd.read_csv('./Input/ks-projects-201801.csv',parse_dates=['deadline', 'launched'])
print(df.state.value_counts())

#lets drop live projects for now

print(df.shape)
df=df.query('state != "live"')
print(df.shape)


# add a columnn that specify the success or unsuccessful projects
df=df.assign(outcome = (df['state'] == 'successful').astype(int) )

print(df.outcome.value_counts())

#CONVERT times
df=df.assign(   hour = df.launched.dt.hour,
                day = df.launched.dt.day,
                month =df.launched.dt.month, 
                year = df.launched.dt.year
)
print(df.head())

#prepare categorical features
from sklearn.preprocessing import LabelEncoder

print(df.info())
cat_features =['category','currency','country']
encoder = LabelEncoder()

encoded = df[cat_features].apply(encoder.fit_transform)
print(encoded.head())

#select columns to merge with encoded columns
selected_cols =['goal', 'hour', 'day', 'month', 'year', 'outcome']
data =pd.concat((df[selected_cols],encoded), axis=1)
print(data.head())


# create validation and training set
valid_fraction = 0.1
valid_size = int(len(data)*valid_fraction)

train=data[:-2*valid_size]
valid=data[-2*valid_size : -valid_size]
test=data[-valid_size:]

#we should ensure that we have target variable in the same size in each bucket
for bucket in (train,valid,test):
    print('fraction is :',100* round(bucket.outcome.mean(),3))



#train the model
import lightgbm as lgb



#!/usr/bin/env python
# coding: utf-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import Series

import warnings
import seaborn as sns

warnings.filterwarnings("ignore")
train = pd.read_csv('Train_SU63ISt.csv')
test = pd.read_csv('Test_0qrQsBZ.csv')
train_copy = train.copy()
test_copy = test.copy()

train.head()

train.columns, test.columns

# lest look at the data
train.dtypes, test.dtypes

train.shape, test.shape

# convert to time
train['Datetime'] = pd.to_datetime(train['Datetime'], format='%d-%m-%Y %H:%M')
test['Datetime'] = pd.to_datetime(test['Datetime'], format='%d-%m-%Y %H:%M')
train_copy['Datetime'] = pd.to_datetime(train_copy['Datetime'], format='%d-%m-%Y %H:%M')
test_copy['Datetime'] = pd.to_datetime(test_copy['Datetime'], format='%d-%m-%Y %H:%M')

train.head()

# lets get y,m,d from each datset
for i in (train, train_copy, test, test_copy):
    i['Year'] = i['Datetime'].dt.year
    i['Month'] = i['Datetime'].dt.month
    i['Day'] = i['Datetime'].dt.day
    i['Hour'] = i['Datetime'].dt.hour

# lets see the result
train.tail()

# ### Lets see the effect of weekend on the traffic


train['weekday'] = train['Datetime'].dt.dayofweek
train['weekend'] = [1 if x in (5, 6) else 0 for x in train['weekday']]

train.boxplot(by='weekend', column='Count', grid=False)

sns.boxplot(x='weekend', y='Count', data=train, showfliers=False)

# ## lets see the trend


train.index = train['Datetime']
train.drop('ID', inplace=True, axis=1)
figure(figsize=(16, 8))
plot(train['Count'], label='Passenger count')
title('Time series Date')
xlabel('Time')
ylabel('Count passengers')
legend(loc='best')
show()

# ### List of Hypotesis:
# ### *Traffic will increase as the years pass by
# ###  *Traffic will be high from May to October
# ###  *Traffic on weekdays will be more
# ###  *Traffic during the peak hours will be high


### *Traffic will increase as the years pass by
train.groupby('Year')['Count'].mean().plot.bar()

# more statistical compariosn!
sns.boxplot(x='Year', y='Count', data=train, showfliers=False)

###  *Traffic will be high from May to October
# train.groupby('Month')['Count'].mean().plot.bar()
# this seems incorrect as the average previous years affects so:
temp = train.groupby(['Year', 'Month'])['Count'].mean()
temp = pd.DataFrame(temp)
temp.plot(figsize=(15, 5), title='Passenger Count(Monthwise)', fontsize=14)

train.groupby('Day')['Count'].mean().plot.bar(color='b')

# it seems days do not have a good insight on the hypothesis so lets see Hours
train.groupby('Hour')['Count'].mean().plot.bar(color='b')

# lets see weekday affect on traffice
train.groupby('weekend')['Count'].mean().plot.bar(color='b')

train.groupby('weekday')['Count'].mean().plot.bar(color='b')

# lets aggreagte times to bigger intervals to have a better insight
hourly = train.resample('H').mean()
Daily = train.resample('D').mean()
Weekly = train.resample('W').mean()
Montly = train.resample('M').mean()

# lets visulaize them
fig, axs = subplots(4, 1)
hourly.Count.plot(figsize=(16, 8), title='Hourly', ax=axs[0])
Daily.Count.plot(figsize=(16, 8), title='Daily', ax=axs[1])
Weekly.Count.plot(figsize=(16, 8), title='Weekly', ax=axs[2])
Montly.Count.plot(figsize=(16, 8), title='Montly', ax=axs[3])

test.Timestamp = pd.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')
test.index = test.Timestamp

# Converting to daily mean 
test = test.resample('D').mean()

# #### lets create validation and Train set
# #### last  month will be in the validation because if we choose randomly in validation set it means predict old data based
# #### on future that is meaningless so for timebased dataset we use this 


Train = train.ix['2012-08-25':'2014-06-24']
Valid = train.ix['2014-06-25':'2014-09-25']

Train.Count.plot(figsize=(15, 8), title='Daily Ridership', fontsize=14, label='train')
Valid.Count.plot(figsize=(15, 8), title='Daily Ridership', fontsize=14, label='Valid')
xlabel('Time')
ylabel('Count')
legend(loc='best')
show()

# ## lets dive into the forecasting models
# #### i) Naive Approach
# #### ii) Moving Average
# #### iii) Simple Exponential Smoothing
# #### iv) Holt’s Linear Trend Model


#### i) Naive Approach
# put the latest count for other points
dd = np.asanyarray(Train.Count)
y_hat = Valid.copy()
y_hat['naive'] = dd[len(dd) - 1]

# lets visualize the data
figure(figsize=(12, 8))
plot(Train.index, Train.Count, label='Train')
plot(Valid.index, Valid.Count, label='Valid')
plot(y_hat.index, y_hat.naive, label='naive')
legend(loc='best')
title("Naive Forecast")
show()

Train.tail()

# lets calculate rsme
from sklearn.metrics import mean_squared_error
from math import sqrt

rsme = sqrt(mean_squared_error(y_hat.Count, y_hat.naive))
print(rsme)

# #### ii) Moving Average
# 


y_hat_avg = Valid.copy()
y_hat_avg['avg_forecast'] = y_hat['Count'].rolling(10).mean().iloc[-1]
rmse = sqrt(mean_squared_error(y_hat.Count, y_hat_avg.avg_forecast))
figure(figsize=(15, 8))
plot(Train.index, Train.Count, label='Train')
plot(Valid.index, Valid.Count, label='Valid')
plot(y_hat_avg.avg_forecast, label='forecast')
legend(loc='Best')
title('moving avg with 10 rolling with rmse:{}'.format(round(rmse, 2)))
show()

y_hat_avg = Valid.copy()
y_hat_avg['avg_forecast'] = y_hat['Count'].rolling(20).mean().iloc[-1]
rmse = sqrt(mean_squared_error(y_hat.Count, y_hat_avg.avg_forecast))
figure(figsize=(15, 8))
plot(Train.index, Train.Count, label='Train')
plot(Valid.index, Valid.Count, label='Valid')
plot(y_hat_avg.avg_forecast, label='forecast')
legend(loc='Best')
title('moving avg with 20 rolling with rmse:{}'.format(round(rmse, 2)))
show()

y_hat_avg = Valid.copy()
y_hat_avg['avg_forecast'] = y_hat['Count'].rolling(50).mean().iloc[-1]
rmse = sqrt(mean_squared_error(y_hat.Count, y_hat_avg.avg_forecast))
figure(figsize=(15, 8))
plot(Train.index, Train.Count, label='Train')
plot(Valid.index, Valid.Count, label='Valid')
plot(y_hat_avg.avg_forecast, label='forecast')
legend(loc='Best')
title('moving avg with 50 rolling with rmse:{}'.format(round(rmse, 2)))
show()

# iii) Simple Exponential Smoothing¶
# 


from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

y_hat_avg = Valid.copy()
fit2 = SimpleExpSmoothing(np.array(Train['Count'])).fit(smoothing_level=0.6, optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(Valid))
rmse = np.round(sqrt(mean_squared_error(Valid['Count'], y_hat_avg['SES'])), 1)

figure(figsize=(16, 8))
plot(Train['Count'], label='Train')
plot(Valid['Count'], label='Valid')
plot(y_hat_avg['SES'], label='SES')
title('SimpleExpSmoothing with RSM:{}'.format(rmse))
legend(loc='Best')
show()

# iv) Holt’s Linear Trend Model


import statsmodels.api as sm

sm.tsa.seasonal_decompose(Train.Count).plot()
result = sm.tsa.adfuller(train.Count)
show()

y_hat_avg = Valid.copy()
fit1 = Holt(np.array(Train.Count)).fit(smoothing_level=0.3)
y_hat_avg['Holter_linear'] = fit1.forecast(len(Valid))

rmse = np.round(sqrt(mean_squared_error(Valid['Count'], y_hat_avg['Holter_linear'])), 1)

figure(figsize=(16, 8))
plot(Train['Count'], label='Train')
plot(Valid['Count'], label='Valid')
plot(y_hat_avg['Holter_linear'], label='Holter_linear')
title('Holter_linear with RSM:{}'.format(rmse))
legend(loc='Best')
show()

y_hat_avg.tail()

submission = pd.read_csv('Sample_Submission_QChS6c3.csv')

# lets predict on test data set
predict = fit1.forecast(len(test))
test['prediction'] = predict

# calculate the ratio of passenger count for each hour of every day
train_copy['ratio'] = train_copy['Count'] / train_copy['Count'].sum()
train_copy.head()

# the average ratio of passenger count for every hour and we will get  ratios
temp = train_copy.groupby('Hour')['ratio'].sum()
temp.head()
# Groupby to csv format 
pd.DataFrame(temp, columns=['Hour', 'ratio']).to_csv('Groupby.csv')
temp2 = pd.read_csv('Groupby.csv')
temp2.drop('Hour.1', 1, inplace=True)
temp2.head()

# Merge Test and test_original on day, month and year
merge = pd.merge(test, test_copy, on=('Day', 'Month', 'Year'), how='left')
merge['Hour'] = merge['Hour_y']
merge.drop(['Year', 'Month', 'Datetime', 'Hour_x', 'Hour_y'], axis=1, inplace=True)
merge.head()

# Predicting by merging merge and temp
prediction = pd.merge(merge, temp2, on='Hour', how='left')
prediction.head()

# Converting the ratio to the original scale
prediction['count'] = prediction['prediction'] * prediction['ratio'] * 24
prediction['ID'] = prediction['ID_y']
prediction.drop(['ID_x', 'Day', 'prediction', 'ratio', 'Hour', 'ID_y'], axis=1, inplace=True)
prediction.head()

# upload data
prediction.to_csv('Holter_linear.csv', columns=['ID', 'count'], index=False)
# Got  rms in leader board


prediction.head()

#  ) Holt winter’s model on daily time series


y_hat_avg = Valid.copy()
fit1 = ExponentialSmoothing(np.array(Train['Count']), seasonal_periods=7, trend='add', seasonal='add').fit()
y_hat_avg['Holter_Winter'] = fit1.forecast(len(Valid))
rmse = np.round(sqrt(mean_squared_error(Valid['Count'], y_hat_avg['Holter_Winter'])), 1)

figure(figsize=(16, 8))
plot(Train['Count'], label='Train')
plot(Valid['Count'], label='Valid')
plot(y_hat_avg['Holter_Winter'], label='Holter_Winter')
title('Holt_Winter with rmse:'.format(rmse))
legend(loc='Best')
show()

#  Expand to the test dataset


predict = fit1.forecast(len(test))
test['prediction'] = predict

test.head()

# Merge Test and test_original on day, month and year
merge = pd.merge(test, test_copy, on=('Day', 'Month', 'Year'), how='left')
merge.head()

merge['Hour'] = merge['Hour_y']
merge.drop(['Year', 'Month', 'Datetime', 'Hour_x', 'Hour_y'], axis=1, inplace=True)
merge.head()

# Predicting by merging merge and temp
prediction = pd.merge(merge, temp2, on='Hour', how='left')
prediction['Count'] = prediction['prediction'] * prediction['ratio'] * 24
prediction.head()

prediction['ID'] = prediction['ID_y']
submission = prediction.drop(['Day', 'Hour', 'ratio', 'prediction', 'ID_x', 'ID_y'], axis=1)
submission.head()

pd.DataFrame(submission, columns=['ID', 'Count']).to_csv('Holt winters.csv', index=False)

# ### Parameter tuning for ARIMA model

from statsmodels.tsa.stattools import adfuller

def test_stationary(timeseries):
    # #Determing rolling statistics
    rolmean=pd.rolling_mean(timeseries,window=24)
    rolstd=pd.rolling_std(timeseries , window = 24)

    #plot the
    orig= plot(timeseries,color='Blue' ,label='Original')
    mean=plot(rolmean , color = 'red' , label = 'rolling mean')
    std=plot(rolstd , color = 'balck' , label='rolling std' )
    legend('best')
    title('Rolling Mean & std ')

    show(block=False)

    # eprform adfuller test
    print('Results of Dickey-Fuller Test:')
    dftest=adfuller(timeseries , autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key , val in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10
test_stationarity(train['Count'])





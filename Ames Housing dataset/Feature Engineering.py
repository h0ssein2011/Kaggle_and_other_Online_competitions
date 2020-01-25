import  pandas as pd

df=pd.read_csv('./Input/ks-projects-201801.csv',parse_dates=['deadline', 'launched'])
print(df.state.value_counts())

#lets drop live projects for now

print(df.shape)
df=df.query('state != "live"')
print(df.shape)


# add a columnn that specify the success or unsuccessful projects
df=df.assign(outcome = (df['state'] == 'successful') )

print(df.outcome.value_counts())

#CONVERT times
df=df.assign(   hour = df.launched.dt.hour,
                day = df.launched.dt.day,
                month =df.launched.dt.month, 
                year = df.launched.dt.year
)
print(df.head())



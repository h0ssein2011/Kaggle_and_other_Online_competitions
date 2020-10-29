import pandas as pd
import numpy as np 

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


for col in train.columns:
    print(col,'has :' , len(train[col].unique()))


print(train.ACTION.value_counts(normalize=True))

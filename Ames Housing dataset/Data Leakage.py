#0- load db

import pandas as pd 
import numpy as np 

card_data = pd.read_csv('Input/AER_credit_card_data.csv',true_values=['yes'],false_values=['no'])
print(card_data.card.value_counts())


#1- select X & y

X=card_data.drop('card',axis = 1 )
y=card_data.card

    #1.1- fill nulls 

#2- find null values
#3- find type vars (int/cats)
#4- preprocessing for each type
#5- build/fit model
#make a pipeline
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
socres = cross_val_score(my_pipeline,X,y,cv=5 ,scoring='accuracy')
# 6- ealuation 
print('cross validation accuracy:{}'.format(socres.mean()))
#7 submuission (if so)

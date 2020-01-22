ask yourself (and also ask data) several basic questions:

### - how does each feature distribution look like? Are there any differences between feature distribution in train and test data?
### - are there any meaningful interactions between the features?
### - are there outliers and can they be explained?
###  -are there missing values or diplicates? What are reasons for them?
And so on.

## Sometimes the results aren't practical - they simply help to understand the data. Often something can be done with the data after EDA.

### - distributions of a feature in train and test are too different? It means that this feature will be hardly useful and it is better to drop it;
### -some feature or target variable has a skewered distribition? It could be useful to transform it (for example logarithm transformation)
### - are outliers normal values or could they be errors? Maybe we should leave them, maybe cut off;
### - plotting mean target value and target value per category it useful. For example there is a feature with 10 categories, we look at mean target values and see that some category has a very high/low mean target value. If there are only several samples in this category, then such high values could be random (then it would be better to combine this category with others) or maybe this category is really important;


# from sklearn import metrics
# from sklearn import tree
# import pandas as pd

# df = pd.read_csv('winequality-red.csv')
# # print(df.head())

# print(df.quality.value_counts(sort=False))

# map_quality = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}
# df.loc[:, 'quality'] = df.quality.map(map_quality)
# print(df.quality.value_counts(sort=False))

# print(df.shape)
# df = df.sample(frac=1).reset_index(drop=True)
# df_train = df.head(1000)
# df_test = df.tail(599)


# clf = tree.DecisionTreeClassifier(max_depth=3)

# cols = [col for col in df.columns if col != 'quality']
# clf.fit(df_train[cols], df_train['quality'])

# # prediction
# train_pred = clf.predict(df_train[cols])
# test_pred = clf.predict(df_test[cols])

# train_accuracy = metrics.accuracy_score(train_pred, df_train.quality)
# test_accuracy = metrics.accuracy_score(test_pred, df_test.quality)

# print('train accuracy:', train_accuracy)
# print('test accuracy:', test_accuracy)


# from sklearn import model_selection
# import pandas as pd

# # kfold cross validation
# if __name__ == '__main__':

#     df = pd.read_csv('winequality-red.csv')
#     df['kfold'] = -1

#     df = df.sample(frac=1).reset_index(drop=True)
#     kf = model_selection.KFold(n_splits=5)

#     for fold, (trn_, val_) in enumerate(kf.split(X=df)):

#         df.loc[val_, 'kfold'] = fold

#     # df.to_csv('train_folds.csv', index=False)

# from sklearn.model_selection import StratifiedKFold
# import pandas as pd
# if __name__ == '__main__':
#     df = pd.read_csv('winequality-red.csv')
#     df = df.sample(frac=1).reset_index(drop=True)
#     df['kfold'] = -1
#     y = df.quality.values
#     skf = StratifiedKFold(n_splits=10)

#     for fold, (trn_, val_) in enumerate(skf.split(X=df, y=y)):
#         df.loc[val_, 'kfold'] = fold

#     print(pd.crosstab(df.quality, df.kfold))

#     # df.to_csv('train_skf.csv', index=False)


# from sklearn import model_selection
# from sklearn import datasets

# import pandas as pd
# import numpy as np


# def create_folds(data):
#     data['fold'] = -1

#     num_bins = int(np.floor(1+np.log2(data.shape[0])))
#     print(num_bins)

#     data.loc[:, 'bins'] = pd.cut(data['target'], bins=num_bins, labels=False)

#     skf = model_selection.StratifiedKFold(n_splits=5)

#     for fold, (trn_, val) in enumerate(skf.split(X=data, y=data.bins.values)):
#         data.loc[val, 'fold'] = fold

#     data = data.drop('bins', axis=1)

#     return data


# if __name__ == '__main__':
#     X, y = datasets.make_regression(
#         n_samples=15000, n_features=100, n_targets=1)

#     df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
#     df.loc[:, 'target'] = y
#     df = create_folds(df)
#     print(df.head())

import numpy as np
import pandas as pd

X = X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

y = np.array([1, 2, 3, 4])

df = pd.DataFrame(X)
df['y'] = y
print(df)

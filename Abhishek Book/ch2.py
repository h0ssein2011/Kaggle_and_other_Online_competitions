import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets, manifold

# # %matplotlib inline


data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)


pixel_values, targets = data
targets = targets.astype(int)
print(pixel_values.shape, targets.shape)

# single_img = pixel_values[1, :].reshape(28, 28)
# plt.imshow(single_img)

tsne = manifold.TSNE(n_components=2, random_state=42)
transformed_data = tsne.fit_transform(pixel_values[:3000, :])

tsne_df = pd.DataFrame(np.column_stack(
    (transformed_data, targets[:3000])),
    columns=['x', 'y', 'targets']
)

tsne_df.loc[:, 'targets'] = tsne_df.targets.astype(int)
print(tsne_df.head())

grid = sns.FacetGrid((tsne_df, hue='targets', size=8)
grid.map(plt.scatter, 'x', 'y').add_legend()


# def tp_pr(y_true, y_pred, threshold=0.5):
#     prcsiosion = []
#     recall = []
#     tp = sum([1 for i in range(len(y_true)) if y_true[i]
#               == 1 and y_pred[i] >= threshold])
#     tn = sum([1 for i in range(len(y_true)) if y_true[i]
#               == 0 and y_pred[i] < threshold])
#     fp = sum([1 for i in range(len(y_true)) if y_true[i]
#               == 0 and y_pred[i] >= threshold])
#     fn = sum([1 for i in range(len(y_true)) if y_true[i]
#               == 1 and y_pred[i] < threshold])
#     # print(tp,tn,fp,fn)
#     prcsiosion.append(tp/(tp+fp))
#     recall.append(tp/(tp + fn))
#     return(prcsiosion, recall)


# print(tp_pr(y_true=[1, 0, 0, 1], y_pred=[0.8, .2, 0.6, 0.7]))

# threshold = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

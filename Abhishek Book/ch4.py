# evaluation metric chapter
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from time import time


def accuracy(y_true, y_pred):

    true_encouter = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            true_encouter += 1
    return true_encouter / len(y_true)


l1 = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0])
l2 = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])

print(f'function acuracy={accuracy(l1, l2)}')
print(f'Sklearn accuracy: {accuracy_score(l1, l2)}')


def true_positive(y_true, y_pred):

    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1

    return tp


def true_negative(y_true, y_pred):

    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1

    return tn


def false_positive(y_true, y_pred):

    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1

    return fp


def false_negative(y_true, y_pred):

    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1

    return fn


def accuracy_ver2(y_true, y_pred):

    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    accuracy_score = (tp + tn) / (tp + tn + fp + fn)
    return accuracy_score


print(f'tp={true_positive(l1,l2)}')
print(f'tn={true_negative(l1,l2)}')
print(f'fp={false_positive(l1,l2)}')
print(f'fn={false_negative(l1,l2)}')
print(f'accuracy_ver2 = {accuracy_ver2(l1,l2)}')


def percision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    percision = tp / (tp + fp)
    return percision


def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    recall = tp / (tp + fn)
    return recall


print(f'percision={percision(l1,l2)}')
print(f'recall={recall(l1,l2)}')

y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

y_pred = [0.02638412, 0.11114267, 0.31620708, 0.0490937, 0.0191491, 0.17554844, 0.15952202, 0.03819563, 0.11639273,
          0.079377, 0.08584789, 0.39095342, 0.27259048, 0.03447096, 0.04644807, 0.03543574, 0.18521942, 0.05934905, 0.61977213, 0.33056815]

thresholds = [0.0490937, 0.05934905, 0.079377, 0.08584789, 0.11114267, 0.11639273, 0.15952202, 0.17554844, 0.18521942, 0.27259048, 0.31620708,
              0.33056815, 0.39095342, 0.61977213]

percisions = []
recalls = []

for i in thresholds:
    temp_pred = [1 if x >= i else 0 for x in y_pred]
    percisions.append(percision(y_true, temp_pred))
    recalls.append(recall(y_true, temp_pred))

print(f'percsions is:{percisions}')
print(f'recalls is:{recalls}')

plt.figure(figsize=(16, 8))
plt.plot(recalls, percisions)
plt.xlabel('recall')
plt.ylabel('percision')
plt.show()


def f1(y_true, y_pred):
    p = percision(y_true, y_pred)
    r = recall(y_true, y_pred)

    score = 2*p * r / (p + r)
    return score


start = time()
print(f'f1 ={f1(l1,l2)}')
print(f'time elapsed ={time() - start}')
start = time()
print(f'f1_sklearn={f1_score(l1,l2)}')
print(f'time elapsed ={time() - start}')


def tpr(y_true, y_pred):
    return recall(y_true, y_pred)


def fpr(y_true, y_pred):
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)

    return fp / (fp + tn)

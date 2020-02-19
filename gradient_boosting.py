import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import numpy as np
import pandas as pd
from functions import *
from sklearn.model_selection import train_test_split
register_matplotlib_converters()

from sklearn.ensemble import GradientBoostingClassifier

import sklearn.metrics as metrics


data: pd.DataFrame = pd.read_csv("data/covtype_processed.csv")

# Preparing data for training
y: np.ndarray = data.pop('Cover_Type').values
X: np.ndarray = data.values
labels: np.ndarray = pd.unique(y)
#0.7 -> train data; 0.3 -> test data
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

def gradient_boosting(trnX, tstX, trnY, tstY):
    learning_rates = [0.1, 0.25, 0.5, 0.75]
    n_estimators = [2, 5, 10, 25]
    max_depths = [2, 5]
    for l in learning_rates:
        for n in n_estimators:
            for m in max_depths:
                gb = GradientBoostingClassifier(n_estimators=n, learning_rate=l, max_features=2,
                                                max_depth=m, random_state=0)
                gb.fit(trnX, trnY)
                prdY = gb.predict(tstX)
                print('Accuracy for %s [learning_rate] with %s [n_estimators] and %s [max_depths] = %.4f' % (l, n, m,
                                                                                    metrics.accuracy_score(tstY, prdY)))

gradient_boosting(trnX, tstX, trnY, tstY)
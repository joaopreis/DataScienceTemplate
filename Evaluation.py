import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pip._vendor.webencodings import labels
from sklearn import metrics
from TrainingModels import *

CMAP = plt.cm.Blues


def plot_confusion_matrix(ax: plt.Axes, cnf_matrix: np.ndarray, classes_names: list, normalize: bool = False):
    if normalize:
        total = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / total
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        title = 'Confusion matrix'
    np.set_printoptions(precision=2)
    tick_marks = np.arange(0, len(classes_names), 1)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cm, interpolation='nearest', cmap=CMAP)
    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center")


##Accuracy##
def accuracy(data, trainSize, className, estimator):
    trnX, tstX, trnY, tstY, labels = trainSplitTest(data, trainSize, className)
    clf = estimator
    clf.fit(trnX, trnY)
    clf.score(tstX, tstY)


##Confusion Matrix##
def confusionMatrix(data, trainSize, className, estimatorName):
    trnX, tstX, trnY, tstY, labels = trainSplitTest(data, trainSize, className)
    clf = estimatorName
    clf.fit(trnX, trnY)
    prdY = clf.predict(tstX)
    plt.figure()
    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
    plot_confusion_matrix(axs[0, 0], metrics.confusion_matrix(tstY, prdY, labels), labels)
    plot_confusion_matrix(axs[0, 1], metrics.confusion_matrix(tstY, prdY, labels), labels, normalize=True)
    plt.tight_layout()
    plt.show()

##ROC Chart##
def plot_roc_chart(ax: plt.Axes, models: dict, tstX, tstY, target: str = 'class'):
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('FP rate')
    ax.set_ylabel('TP rate')
    ax.set_title('ROC chart for %s' % target)
    ax.plot([0, 1], [0, 1], color='navy', label='random', linestyle='--')

    for clf in models:
        scores = models[clf].predict_proba(tstX)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(tstY, scores, 'positive')
        roc_auc = metrics.roc_auc_score(tstY, scores)
        ax.plot(fpr, tpr, label='%s (auc=%0.2f)' % (clf, roc_auc))
    ax.legend(loc="lower center")


def rocChart(data,trainSize,className,estimator):
    y = data.pop(className).values
    X = data.values
    trnX, tstX, trnY, tstY = train_test_split(X, y, trainSize, stratify=y)
    model = estimator.fit(trnX, trnY)
    plt.figure()
    plot_roc_chart(plt.gca(), {'GaussianNB': model}, tstX, tstY, 'class')
    plt.show()

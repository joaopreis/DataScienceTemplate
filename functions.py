import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.tree import export_graphviz

#plt.colorbar()
CMAP = plt.cm.Pastel2
plt.set_cmap(CMAP)
linestyles = ['-', '-.', '--', ':', ' ', '']

def choose_grid(nr):
    return (nr // 4, 4) if nr % 4 == 0 else (nr // 4 + 1, 4)


def line_chart(ax: plt.Axes, series: pd.Series, title: str, xlabel: str, ylabel: str, percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(series.index)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.plot(series, color='black')


def multiple_line_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str,
                        percentage=False):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel, horizontalalignment='right', x=1.0)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xvalues)
    if percentage:
        ax.set_ylim(0.7, 1.0)
    i = 0
    for name, y in yvalues.items():
        ax.plot(xvalues, y, linestyle=linestyles[i], color='black')
        i += 1
        legend.append(name)
    ax.legend(legend, loc='lower center', ncol=len(yvalues), bbox_to_anchor=(0.5, -0.02))


def bar_chart(ax: plt.Axes, xvalues: list, yvalues: list, title: str, xlabel: str, ylabel: str, percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xvalues, rotation=90, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.bar(xvalues, yvalues, edgecolor='grey')


def multiple_bar_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str,
                       percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x = np.arange(len(xvalues))  # the label locations
    ax.set_xticks(x)
    ax.set_xticklabels(xvalues, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    width = 0.8  # the width of the bars
    step = width / len(yvalues)
    i = 0
    for name, y in yvalues.items():
        ax.bar(x + i * step, y, step, label=name)
        i += 1
    ax.legend(loc='lower center', ncol=len(yvalues), bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True)


def plot_confusion_matrix(ax: plt.Axes, cnf_matrix: np.ndarray, classes_names: list, normalize: bool = False):
    if normalize:
        total = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / total
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        title = 'Confusion matrix, without normalization'
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


def plot_roc_chart(ax: plt.Axes, models: dict, tstX: np.ndarray, tstY: np.ndarray, minor, target: str = 'class'):
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('FP rate', horizontalalignment='right', x=1.0)
    ax.set_ylabel('TP rate')
    ax.set_title('ROC chart for %s' % target)
    #ax.plot([0, 1], [0, 1], color='navy', label='random', linestyle='--')
    i = 0
    for clf in models:
        scores = models[clf].predict_proba(tstX)[:, 1]
        roc_auc = metrics.roc_auc_score(tstY, scores)
        print(clf, roc_auc)
        fpr, tpr, _ = metrics.roc_curve(tstY, scores, minor)
        ax.plot(fpr, tpr, label=clf, linestyle=linestyles[i], color='black')
        i += 1
    ax.legend(loc="lower left", ncol=len(models), bbox_to_anchor=(0, -0.2))


def plot_tree(tree, title):
    dot_data = export_graphviz(tree, out_file='dtree.dot', filled=True, rounded=True, special_characters=True)
    # Convert to png
    from subprocess import call
    call(['dot', '-Tpng', 'dtree.dot', '-o', 'dtree.png', '-Gdpi=600'])

    plt.figure(figsize=(14, 18))
    plt.title(title)
    plt.imshow(plt.imread('dtree.png'))
    plt.axis('off')


##python -m pip install --upgrade pip

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as _stats
from functions import *


##CONTENT IDENTIFICATION##
##Read a csv file##
def readCSV(fileName):
    data = pd.read_csv(fileName)
    return data


##Data shape##
def dataShape(data):
    return data.shape


########################################################################################################################


##SINGLE VARIABLE ANALYSIS##
##Select a sample of a data set##
def sampleData(data, frac):
    sample = data.sample(frac=frac)
    return sample


##Data head##
def dataHead(data):
    return data.head()


##Attributes type##
def dataType(data):
    return data.dtypes


##Values of categorical attributes##
def categoricalValues(data, type):
    cat_vars = data.select_dtypes(include=type)
    return cat_vars


##Print categorical attributes##
def printCatValues(data, type):
    cat_vars = categoricalValues(data, type)
    for att in cat_vars:
        print(att, data[att].unique())


##Change attribute type##
def changeType(data, oldType, newType):
    cat_vars = categoricalValues(data, oldType)
    data[cat_vars.columns] = data.select_dtypes([oldType]).apply(lambda x: x.astype(newType))
    return data


##Descriptive statistics##
def describeData(data):
    return data.describe()


##Null values frequency##
def nullValues(data, figS1, figS2):
    fig = plt.figure(figsize=(figS1, figS2))
    mv = {}
    for var in data:
        mv[var] = data[var].isna().sum()
        bar_chart(plt.gca(), mv.keys(), mv.values(), 'Number of missing values per variable', var,
                       'nr. missing values')
    fig.tight_layout()
    plt.show()


##Data boxplot##
def boxplot(data, rot, fontS, figS1, figS2):
    data.boxplot(rot=rot, fontsize=fontS, figsize=(figS1, figS2))
    plt.show()


##Individual boxplot##
def indBoxplot(data, figS1, figS2, start, stop):
    columns = data.select_dtypes(include='number').columns
    rows, cols = choose_grid(stop - start + 1)
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * figS1, rows * figS2), squeeze=False)
    i, j = 0, 0
    c = -1
    for n in range(start, stop + 1):
        axs[i, j].set_title('Boxplot for %s' % columns[n])
        axs[i, j].boxplot(data[columns[n]].dropna().values)
        c = c + 1
        i, j = (i + 1, 0) if (c + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()
    plt.show()


##Individual histogram numeric attributes##
def indNumHistogram(data, figS1, figS2, start, stop):
    columns = data.select_dtypes(include='number').columns
    rows, cols = choose_grid(stop - start + 1)
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * figS1, rows * figS2), squeeze=False)
    i, j = 0, 0
    c = -1
    for n in range(start, stop + 1):
        axs[i, j].set_title('Histogram for %s' % columns[n])
        axs[i, j].set_xlabel(columns[n])
        axs[i, j].set_ylabel("frequency")
        axs[i, j].hist(data[columns[n]].dropna().values, 'auto')
        c = c + 1
        i, j = (i + 1, 0) if (c + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()
    plt.show()


##Individual histogram categorical attributes##
def indCatHistogram(data, category, figS1, figS2, start, stop):
    columns = data.select_dtypes(include=category).columns
    rows, cols = choose_grid(stop - start + 1)
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * figS1, rows * figS2), squeeze=False)
    i, j = 0, 0
    c = -1
    for n in range(start, stop + 1):
        counts = data[columns[n]].dropna().value_counts(normalize=True)
        bar_chart(axs[i, j], counts.index, counts.values, 'Histogram for %s' % columns[n], columns[n],
                       'probability')
        c = c + 1
        i, j = (i + 1, 0) if (c + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()
    plt.show()


##Individual histogram with seaborn##
def indSnsHistogram(data, figS1, figS2, start, stop):
    columns = data.select_dtypes(include='number').columns
    rows, cols = choose_grid(stop - start + 1)
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * figS1, rows * figS2), squeeze=False)
    i, j = 0, 0
    c = -1
    for n in range(start, stop + 1):
        axs[i, j].set_title('Histogram with trend for %s' % columns[n])
        axs[i, j].set_ylabel("probability")
        sns.distplot(data[columns[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=columns[n])
        c = c + 1
        i, j = (i + 1, 0) if (c + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()
    plt.show()


##Histogram with best fit###############################################################################################

def compute_known_distributions(x_values, n_bins) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = _stats.norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)' % (mean, sigma)] = _stats.norm.pdf(x_values, mean, sigma)
    # LogNorm
    #  sigma, loc, scale = _stats.lognorm.fit(x_values)
    #  distributions['LogNor(%.1f,%.2f)'%(np.log(scale),sigma)] = _stats.lognorm.pdf(x_values, sigma, loc, scale)
    # Exponential
    loc, scale = _stats.expon.fit(x_values)
    distributions['Exp(%.2f)' % (1 / scale)] = _stats.expon.pdf(x_values, loc, scale)
    # SkewNorm
    # a, loc, scale = _stats.skewnorm.fit(x_values)
    # distributions['SkewNorm(%.2f)'%a] = _stats.skewnorm.pdf(x_values, a, loc, scale)
    return distributions


def histogram_with_distributions(ax: plt.Axes, series: pd.Series, var: str):
    values = series.sort_values().values
    n, bins, patches = ax.hist(values, 20, density=True, edgecolor='grey')
    distributions = compute_known_distributions(values, bins)
    multiple_line_chart(ax, values, distributions, 'Best fit for %s' % var, var, 'probability')


def indHistDist(data, figS1, figS2, start, stop):
    columns = data.select_dtypes(include='number').columns
    rows, cols = choose_grid(stop - start + 1)
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * figS1, rows * figS2), squeeze=False)
    i, j = 0, 0
    c = -1
    for n in range(start, stop + 1):
        histogram_with_distributions(axs[i, j], data[columns[n]].dropna(), columns[n])
        c = c + 1
        i, j = (i + 1, 0) if (c + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()
    plt.show()


def HistDist(data, figS1, figS2, start, stop):
    columns = data.select_dtypes(include='number').columns
    rows, cols = 1, 1
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * figS1, rows * figS2), squeeze=False)
    for n in range(start, stop + 1):
        histogram_with_distributions(axs[0, 0], data[columns[n]].dropna(), columns[n])
    fig.tight_layout()
    plt.show()


########################################################################################################################

##Histogram with different granularity##
def granularHist(data, figS1, figS2, start, stop):
    columns = data.select_dtypes(include='number').columns
    rows = stop - start
    cols = 5
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * figS1, rows * figS2), squeeze=False)
    bins = range(5, 100, 20)
    for i in range(rows):
        for j in range(len(bins)):
            axs[i, j].set_title('Histogram for %s' % columns[i])
            axs[i, j].set_xlabel(columns[i])
            axs[i, j].set_ylabel("probability")
            axs[i, j].hist(data[columns[i]].dropna().values, bins[j])
    fig.tight_layout()
    plt.show()


########################################################################################################################

##MULTIPLE VARIABLE ANALYSIS##
##Coorrelation analysis##
def heatMap(data, figS1, figS2, start, stop):
    fig = plt.figure(figsize=[figS1, figS2])
    corr_mtx = data.iloc[:, start:stop + 1].corr()
    sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    plt.title('Correlation analysis')
    plt.show()


##Scatter Plot##
def sparsity(data):
    columns = data.select_dtypes(include='number').columns
    rows, cols = len(columns) - 1, len(columns) - 1
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    for i in range(len(columns)):
        var1 = columns[i]
        for j in range(i + 1, len(columns)):
            var2 = columns[j]
            axs[i, j - 1].set_title("%s x %s" % (var1, var2))
            axs[i, j - 1].set_xlabel(var1)
            axs[i, j - 1].set_ylabel(var2)
            axs[i, j - 1].scatter(data[var1], data[var2])
    fig.tight_layout()
    plt.show()

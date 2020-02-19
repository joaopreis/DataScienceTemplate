from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pydot
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from subprocess import call
from subprocess import check_call

from functions import *


##Training strategies##

##k-fold cross validation (StratifiedKFold): used in the presence of a few thousand records;
##hold-out (train_test_split): used in the presence of large thousands of records;
##sample hold-out: used in the presence of millions of records.


def trainSplitTestLabel(data, trainSize, className):
    y: np.ndarray = data.pop(className).values
    X: np.ndarray = data.values
    labels: np.ndarray = pd.unique(y)
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=trainSize, stratify=y)
    return trnX, tstX, trnY, tstY, labels


def trainSplitTest(data, trainSize, className):
    y: np.ndarray = data.pop(className).values
    X: np.ndarray = data.values
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=trainSize, stratify=y)
    return trnX, tstX, trnY, tstY


##Naive Bayes##
def naiveBayes(data, trainSize, className, estimator):
    trnX, tstX, trnY, tstY, labels = trainSplitTestLabel(data, trainSize, className)
    clf = estimator
    clf.fit(trnX, trnY)
    prdY = clf.predict(tstX)
    cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)
    plot_confusion_matrix(plt.gca(), cnf_mtx, labels)
    print("Model accuracy: " + "%.3f" % clf.score(tstX, tstY))
    plt.show()


##Naive Bayes estimators comparison##
def naiveBayesComparator(data, trainSize, className):
    estimators = {'GaussianNB': GaussianNB(), 'BernoulyNB': BernoulliNB()}  ##'MultinomialNB': MultinomialNB()
    trnX, tstX, trnY, tstY = trainSplitTest(data, trainSize, className)
    xvalues = []
    yvalues = []
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        yvalues.append(metrics.accuracy_score(tstY, prdY))
        print("Model accuracy" + clf + ": " + "%.3f" % estimators[clf].score(tstX, tstY))
    plt.figure()
    bar_chart(plt.gca(), xvalues, yvalues, 'Comparison of Naive Bayes Models', '', 'accuracy', percentage=True)
    plt.show()


##KNN##
def kNNComparator(data, trainSize, className):
    trnX, tstX, trnY, tstY = trainSplitTest(data, trainSize, className)
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    for d in dist:
        yvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            print("Model accuracy " + d + "(" + "%.3f" % n + "): " + "%.3f" % metrics.accuracy_score(tstY, prdY))
            yvalues.append(metrics.accuracy_score(tstY, prdY))
        values[d] = yvalues
    plt.figure()
    multiple_line_chart(plt.gca(), nvalues, values, 'KNN variants', 'n', 'accuracy', percentage=True)
    plt.show()


##Decision Trees##
def decisionTrees(data, trainSize, className):
    min_samples_leaf = [.05, .025, .01, .0075, .005, .0025, .001]
    max_depths = [5, 10, 25, 50]
    criteria = ['entropy', 'gini']
    trnX, tstX, trnY, tstY = trainSplitTest(data, trainSize, className)
    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for n in min_samples_leaf:
                tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f)
                tree.fit(trnX, trnY)
                prdY = tree.predict(tstX)
                print("Model accuracy " + "%.3f" % d + "(" + "%.3f" % n + "): " + "%.3f" % metrics.accuracy_score(tstY,
                                                                                                                  prdY))
                yvalues.append(metrics.accuracy_score(tstY, prdY))
            values[d] = yvalues
        multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s criteria' % f,
                            'nr estimators',
                            'accuracy', percentage=True)
    plt.show()


##TreePNG##
##Tree PNG##
def treePNG(data, trainSize, maxDepth, className):
    tree = DecisionTreeClassifier(max_depth=maxDepth)
    trnX, tstX, trnY, tstY = trainSplitTest(data, trainSize, className)
    tree.fit(trnX, trnY)
    dot_data = export_graphviz(tree, out_file='dtree.dot', filled=True, rounded=True, special_characters=True)
    plt.figure(figsize=(14, 18))
    plt.axis('off')
    (graph,) = pydot.graph_from_dot_file('dtree.dot')
    graph.write_png('structure2.png')
    plt.show()


##Random forests##
def randomForests(data, trainSize, className):
    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25, 50]
    max_features = ['sqrt', 'log2']
    trnX, tstX, trnY, tstY = trainSplitTest(data, trainSize, className)
    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for k in range(len(max_features)):
        f = max_features[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for n in n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(trnX, trnY)
                prdY = rf.predict(tstX)
                print("Model accuracy " + "%.3f" % d + "(" + "%.3f" % n + "): " + "%.3f" % metrics.accuracy_score(tstY,
                                                                                                                  prdY))
                yvalues.append(metrics.accuracy_score(tstY, prdY))
            values[d] = yvalues
        multiple_line_chart(axs[0, k], n_estimators, values, 'Random Forests with %s features' % f,
                            'nr estimators',
                            'accuracy', percentage=True)
    plt.show()


##Gradient Boosting##
def gradient_boosting(data, trainsSize, className):
    trnX, tstX, trnY, tstY = trainSplitTest(data, trainsSize, className)
    learning_rates = [0.1, 0.25, 0.5, 0.75]
    n_estimators = [2, 5, 10, 25, 50, 75, 100]
    max_depths = [2, 5]
    for l in learning_rates:
        for n in n_estimators:
            for m in max_depths:
                gb = GradientBoostingClassifier(n_estimators=n, learning_rate=l, max_features=2,
                                                max_depth=m, random_state=0)
                gb.fit(trnX, trnY)
                prdY = gb.predict(tstX)
                print('Accuracy for %s [learning_rate] with %s [n_estimators] and %s [max_depths] = %.4f' % (l, n, m,
                                                                                                             metrics.accuracy_score(
                                                                                                                 tstY,
                                                                                                                 prdY)))

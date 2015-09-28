import unittest
from classifier import RandomForest_scikit, SVM_scikit
from FeatureSelection import FeatureSelectionScikit

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.cross_validation import StratifiedKFold

from sklearn import preprocessing

from sklearn.feature_selection import RFE, VarianceThreshold

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn import decomposition
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from time import time

import pickle
import csv
from itertools import izip

def open_model(name):
    archive = open(name, 'r')
    models = pickle.load(archive)
    return models[0], models[1]


class TestStringMethods(unittest.TestCase):
    def upper(self):
        model = FeatureSelectionScikit()
        rdmForestPre = RandomForest_scikit()
        rdmForest = RandomForest_scikit()
        file = open("models.obj", 'r')
        models = pickle.load(file)
        samples = models[0]
        responses = models[1]

        '''
        pca = decomposition.PCA()
        X_digits = samples
        y_digits = responses

        pca.fit(X_digits)

        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.axes([.2, .2, .7, .7])
        plt.plot(pca.explained_variance_, linewidth=2)
        plt.axis('tight')
        plt.xlabel('n_components')
        plt.ylabel('explained_variance_')
        plt.show()
        '''

        # Scaled data
        #samplesScaled = preprocessing.scale(samples)
        samplesScaled = samples

        model.fit(samplesScaled, responses)
        variablesImportance = model.importance()
        mean = np.mean(variablesImportance)
        std = np.std(variablesImportance)

        fig1 = plt.figure(1, figsize=(4, 3))
        ax1 = fig1.add_subplot(111)
        ax1.plot(variablesImportance, linewidth=2)

        basicPre = []
        indices = []
        minimo = min(variablesImportance)

        for i, value in enumerate(variablesImportance):
            if value > minimo:
                basicPre.append(value)
                indices.append(i)

        print('Escogi %d' % (len(basicPre)))

        fig2 = plt.figure(2, figsize=(4, 3))
        ax2 = fig2.add_subplot(111)
        ax2.plot(basicPre, linewidth=2)

        newSample = []
        for i, fila in enumerate(samplesScaled):
            newSample.append([val for is_good, val in izip(indices, fila) if is_good])

        t0 = time()
        rdmForestPre.train(newSample, responses)
        a, confusionPre = rdmForestPre.test(newSample, responses, True)
        preTiempo = (time() - t0)
        print("With Preprocessing %0.3fs" % (preTiempo))

        sumPre = 0
        for idx, fila in enumerate(confusionPre):
            for jdx, entrada in enumerate(fila):
                if idx != jdx:
                    sumPre += entrada

        t0 = time()
        rdmForest.train(samples, responses)
        a, confusion = rdmForest.test(samples, responses, True)
        Tiempo = time() - t0
        print("Without Preprocessing %0.3fs" % (Tiempo))
        print("Preprocessing/Without = %0.3fs" % (1.0 * preTiempo / Tiempo))

        sum = 0
        for idx, fila in enumerate(confusion):
            for jdx, entrada in enumerate(fila):
                if idx != jdx:
                    sum += entrada

        print(str(sumPre), str(sum), float(1.0 * sumPre / sum))

        plt.show()

    def feature_selection_extra_trees(self):
        # Load Data
        file = open("models.obj", 'r')
        models = pickle.load(file)
        samples = models[0]
        responses = models[1]

        # Using ExtraTreesClassifier

        forest = FeatureSelectionScikit(n_estimators=10, criterion="gini")
        forest.fit(samples=samples, response=responses)
        importances = forest.importance()
        std = np.std([tree.feature_importances_ for tree in forest.model.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        to_plot = []
        to_indices = []
        for f in range(50):
            to_plot.append(importances[indices[f]])
            to_indices.append(indices[f])
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(50), to_plot, color="b", yerr=std[to_indices], align="center")
        locs, labels = plt.xticks(range(50), indices[50:])
        plt.setp(labels, rotation=90)
        plt.xlim([-1, 50])
        plt.show()

    def feature_selection_RFE(self):
        # TODO Use accurate model
        # Load Data
        file = open("models.obj", 'r')
        models = pickle.load(file)
        samples = models[0]
        responses = models[1]

        #model = RandomForest_scikit()
        model = RandomForestClassifier()

        selection = RFE(model, 20)
        selection = selection.fit(samples, responses)

        print(selection.support_)
        print(selection.ranking_)

    def feature_selection_variance(self):
        # STATUS Works
        samples, responses = open_model("models.obj")
        selection = VarianceThreshold(threshold=0.5)
        selection.fit_transform(samples)

    def selection_variance_random_tree_k_fold(self):
        # Feature Selection
        samples, responses = open_model("models.obj")
        samples = np.array(samples)
        responses = np.array(responses)

        FeatureSelection = True

        if FeatureSelection:
            selection = VarianceThreshold(threshold=0.00)
            selection.fit(samples)
            idxs = selection.get_support(indices=True)
            samples = samples[:, idxs]

        samples = preprocessing.scale(samples)

        # Stratified cross-validation
        scv = StratifiedKFold(responses, n_folds=10)
        sum = 0
        for i, (train, test) in enumerate(scv):
            print('Case %d' % (i))
            # Modeling
            rdmForest = RandomForest_scikit()

            # Train
            init = time()
            rdmForest.train(samples[train, :], responses[train])

            # Test
            a, confusionPre = rdmForest.test(samples[test, :], responses[test], True)
            print('Time: %0.3fs' % (time() - init))

            for idx, fila in enumerate(confusionPre):
                for jdx, entrada in enumerate(fila):
                    if idx != jdx:
                        sum += entrada

        print("Wrong Cases: "+str(sum))
        print(' Full Case ')
        rdmForest = RandomForest_scikit()
        rdmForest.train(samples, responses)
        rdmForest.test(samples, responses, True)

    def test_variance_k_best_random_tree_k_fold(self):
        # Feature Selection
        samples, responses = open_model("models.obj")
        samples = np.array(samples)
        responses = np.array(responses)

        FeatureSelection = True

        if FeatureSelection:
            selection = VarianceThreshold(threshold=0.00)
            selection.fit(samples)
            idxs = selection.get_support(indices=True)
            samples = samples[:, idxs]

        samples = preprocessing.scale(samples)

        # Stratified cross-validation
        scv = StratifiedKFold(responses, n_folds=10)
        sum = 0
        for i, (train, test) in enumerate(scv):
            print('Case %d' % (i))
            # Modeling
            rdmForest = RandomForest_scikit()

            # Train
            init = time()
            rdmForest.train(samples[train, :], responses[train])

            # Test
            a, confusionPre = rdmForest.test(samples[test, :], responses[test], True)
            print('Time: %0.3fs' % (time() - init))

            for idx, fila in enumerate(confusionPre):
                for jdx, entrada in enumerate(fila):
                    if idx != jdx:
                        sum += entrada

        print("Wrong Cases: "+str(sum))
        print(' Full Case ')
        rdmForest = RandomForest_scikit()
        rdmForest.train(samples, responses)
        rdmForest.test(samples, responses, True)

if __name__ == '__main__':
    unittest.main()

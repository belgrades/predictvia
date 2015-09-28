#from GBDTrees import gbd_trees
from random import shuffle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
from sklearn import svm
import scipy.stats as stats
import numpy as np
import pickle

from utils.general import log


class RandomForest_scikit(object):

    def __init__(self, estimators = None):
        # default params
        if estimators:
            self.model = RandomForestClassifier(n_estimators=estimators)
        else:
            self.model = RandomForestClassifier()

    def train(self, samples, responses):
        self.model.fit(samples, responses)

    def predict(self, samples):
        return self.model.predict(samples)

    def test(self, samples, labels, print_=False):
        prediction = self.model.predict(samples)
        if print_:
            log(metrics.classification_report(labels, prediction))
            log(metrics.confusion_matrix(labels, prediction))
        return metrics.precision_recall_fscore_support(labels, prediction),\
               metrics.confusion_matrix(labels, prediction)

    def proba(self, samples):
        return self.model.predict_proba(samples)

    def final_class_proba(self, samples):
        matrix_proba = self.proba(samples)
        return np.amax(matrix_proba, axis=1)


class SVM_scikit(object):

    def __init__(self):
        self.model = svm.SVC(kernel='linear', gamma=0.7, C=1)

    def train(self, samples, responses):
        self.model.fit(samples, responses)

    def predict(self, samples):
        return self.model.predict(samples)

    def train(self, samples, labels, print_=False):
        prediction = self.model.predict(samples)
        if print_:
            log(metrics.classification_report(labels, prediction))
            log(metrics.confusion_matrix(labels, prediction))
        return metrics.precision_recall_fscore_support(labels, prediction),\
               metrics.confusion_matrix(labels, prediction)


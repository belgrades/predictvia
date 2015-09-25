import unittest
from classifier import RandomForest_scikit, SVM_scikit, FeatureSelection_scikit
from sklearn import preprocessing

from sklearn import decomposition
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from time import time

import pickle
import csv
from itertools import izip

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        model = FeatureSelection_scikit()
        rdmForestPre = RandomForest_scikit()
        rdmForest = RandomForest_scikit()
        file = open("models.obj", 'r')
        models = pickle.load(file)
        samples = models[0]
        responses= models[1]

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
        samplesScaled = preprocessing.scale(samples)

        model.fit(samplesScaled, responses)
        variablesImportance = model.importance()
        mean = np.mean(variablesImportance)

        fig1 = plt.figure(1, figsize=(4, 3))
        ax1 = fig1.add_subplot(111)
        ax1.plot(variablesImportance, linewidth=2)

        basicPre = []
        indices = []
        for i, value in enumerate(variablesImportance):
            if value > mean:
                basicPre.append(value)
                indices.append(i)


        fig2 = plt.figure(2, figsize=(4, 3))
        ax2 = fig2.add_subplot(111)
        ax2.plot(basicPre, linewidth=2)

        newSample = []
        for i, fila in enumerate(samplesScaled):
            newSample.append([val for is_good, val in izip(indices, fila) if is_good])

        t0 = time()
        rdmForestPre.train(newSample, responses)
        a, confusionPre = rdmForestPre.test(newSample, responses, True)
        print("With Preprocessing %0.3fs" % (time() - t0))

        sumPre = 0
        for idx, fila in enumerate(confusionPre):
            for jdx, entrada in enumerate(fila):
                if idx != jdx:
                    sumPre += entrada

        t0 = time()
        rdmForest.train(samples, responses)
        a, confusion = rdmForest.test(samples, responses, True)
        print("Without Preprocessing %0.3fs" % (time() - t0))

        sum = 0
        for idx, fila in enumerate(confusion):
            for jdx, entrada in enumerate(fila):
                if idx != jdx:
                    sum += entrada

        print(str(sumPre), str(sum), float(1.0*sumPre/sum))

        plt.show()


    def PCA_SVM(self):
        svm = SVM_scikit()
        file = open("models.obj", 'r')
        models = pickle.load(file)
        samples = models[0]
        responses = models[1]
        svm.train(samples, responses)
        #svm.test(samples, responses, True)



if __name__ == '__main__':
    unittest.main()
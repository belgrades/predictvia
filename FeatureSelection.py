from utils.general import log
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
import scipy.stats as stats


class FeatureSelectionScikit(object):

    def __init__(self, n_estimators=10, criterion="gini", max_features="auto"):
        self.model = ExtraTreesClassifier(n_estimators=n_estimators, criterion=criterion, max_features=max_features)

    def fit(self, samples, response):
        self.model.fit(samples, response)

    def importance(self):
        log(stats.describe(self.model.feature_importances_))
        return self.model.feature_importances_


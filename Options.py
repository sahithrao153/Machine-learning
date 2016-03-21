import numpy as np


class Parameters(object):
    @classmethod
    def Mean_(self, X):
        return np.mean(X.astype(float), axis=0)

    @classmethod
    def Standard_deviation(self, X):
        return np.std(X.astype(float), axis=0)

    @classmethod
    def Covariance(self, X):
        return np.cov(X.astype(float).T)

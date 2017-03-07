# -*- coding: utf-8 -*-

import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

def knn_model_test(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print ("score = %s" % (np.mean(y_pred == y_test)))


def main(argv):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris['data'], iris['target'], random_state=0)

    knn_model_test(X_train, X_test, y_train, y_test)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

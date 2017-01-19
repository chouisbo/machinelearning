# scikit-learn multi linear regression example

from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

dataPath = "../data/DeliveryDummy.csv"
deliveryData = genfromtxt(dataPath, delimiter=',')

print("data:", deliveryData)

X, Y = deliveryData[:,:-1], deliveryData[:, -1]

print("X:", X)
print("Y:", Y)

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print("coefficients:", regr.coef_)
print("intercept:", regr.intercept_)


xPred = [102, 6, 1, 0, 0]
yPred = regr.predict(xPred)
print("xPred:", xPred, " => yPred:", yPred)


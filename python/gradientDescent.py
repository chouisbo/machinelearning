#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import random


# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d => Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta


def genData(numPoints, bias, variance):
    x = np.zeros(shape = (numPoints, 2))
    y = np.zeros(shape = numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0], x[i][1] = 1, i
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y


if __name__ == "__main__":
    np.random.seed(871109)

    x, y = genData(100, 25, 10)
    print("x:", x)
    print("y:", y)
    m, n = np.shape(x)
    numIterations, alpha, theta = 100000, 0.0005, np.ones(n)
    theta = gradientDescent(x, y, theta, alpha, m, numIterations)
    print("theta:", theta)
    

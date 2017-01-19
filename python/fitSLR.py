# Simple Linear Regression

import numpy as np

def fitSLR(x, y):
    n, dinominator, numerator = len(x), 0, 0
    x_mean, y_mean = np.mean(x), np.mean(y)
    for i in range(0, n):
    	numerator += (x[i] - x_mean)*(y[i] - y_mean)
    	dinominator += (x[i] - x_mean)**2
    b1 = numerator / float(dinominator)
    b0 = y_mean - b1 * x_mean
    return b0, b1

def predict(x, b0, b1):
    return b0 + x*b1

x = [1, 3, 2, 1, 3]
y = [14, 24, 18, 17, 27]    

b0, b1 = fitSLR(x, y)

print("intercept:", b0, " slope:", b1)

x_test = 6

y_test = predict(6, b0, b1)

print("y_test:", y_test)


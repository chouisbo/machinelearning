#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    # return 1.0 - np.tanh(x) * np.tanh(x)
    return 1.0 - x ** 2


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def logistic_derivative(x):
    # return logistic(x) * (1 - logistic(x))
    return x * (1.0 - x)


class NeuralNetwork(object):
    def __init__(self, layers, activation='tanh', weights = None, bias = None):
        """
        :param layers: A list containing the number of units in each layer
        Should be at least two values
        :param activation: The activation function to be used. Can be 
        'logistic' or 'tanh'
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh 
            self.activation_deriv = tanh_derivative 

        if weights != None:
            self.weights = weights
        else:
            self.weights = []
            for i in range(0, len(layers) - 1):
                self.weights.append((2 * np.random.random((layers[i], layers[i+1])) - 1) * 0.25)

        if bias != None:
            self.bias = bias
        else:
            self.bias = []
            for i in range(1, len(layers)):
                self.bias.append((2 * np.random.random((1, layers[i])) - 1) * 0.25)


    def fit(self, X, y, learning_rate=.2, epochs=10000):
        print("weights:", self.weights)
        print("bias:", self.bias)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            # print("randomly pick up one input:", a)

            # going forward network for each layer 
            for l in range(len(self.weights)):
                # compute the node value for each layer(O_i) using activation function
                a.append(self.activation(np.dot(a[l], self.weights[l]) + self.bias[l]))
            # print("forword:", a)
            
            # output layer, err calculation (delta is updated error)
            deltas = [(y[i] - a[-1]) * self.activation_deriv(a[-1])]
            # starting backprobagation
            # begin at the second to last layer
            for l in range(len(a) - 2, 0, -1):
                #Compute the updated error (i,e, deltas) for each node going from top layer to input layer
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            # print("deltas:", deltas)

            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
                self.bias[i] += learning_rate * delta
            # print("updated weights:", self.weights)
            # print("updated bias:", self.bias)
        print("final weights:", self.weights)
        print("final bias:", self.bias)


    def predict(self, X):
        x = np.array(X)
        for l in range(0, len(self.weights)):
            x = self.activation(np.dot(x, self.weights[l]) + self.bias[l])
        return x


def test_example():
    np.random.seed(16319)
    weights = [ np.array([[.2, -.3], [.4, .1], [-.5, .2]]),
                np.array([[-.3], [-.2]]) ]
    bias = [ np.array([[-.4, .2]]), np.array([[.1]])]
    nn = NeuralNetwork([3, 2, 1], activation = 'logistic', weights = weights, bias = bias)
    X, y = np.array([[1, 0, 1]]), np.array([1])
    nn.fit(X, y, learning_rate=.9, epochs=1)


def test_function_XOR():
    np.random.seed(1208)
    nn = NeuralNetwork([2, 2, 1], activation = 'tanh')
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    nn.fit(X, Y)
    for record in X:
        print(record, nn.predict(record))


if __name__ == "__main__":
    # test_example()
    test_function_XOR()


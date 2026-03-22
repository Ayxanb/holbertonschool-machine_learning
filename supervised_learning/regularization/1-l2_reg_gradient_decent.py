#!/usr/bin/env python3
"""
This module contains a function to update weights and biases
using gradient descent with L2 regularization.
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization.

    Args:
        Y: one-hot numpy.ndarray (classes, m) containing correct labels
        weights: dictionary of weights and biases
        cache: dictionary of the outputs of each layer
        alpha: the learning rate
        lambtha: the L2 regularization parameter
        L: the number of layers of the network
    """
    m = Y.shape[1]
    # Backpropagation starts from the output layer
    # dZ for softmax output layer (A_L - Y)
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        # Calculate gradients
        # dW includes the L2 penalty term: (lambda / m) * W
        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            # Calculate dZ for the next layer (tanh activation)
            # dZ = W.T * dZ * g'(Z), where g'(Z) = 1 - A^2
            dZ = np.dot(W.T, dZ) * (1 - np.power(A_prev, 2))

        # Update weights and biases in place
        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db

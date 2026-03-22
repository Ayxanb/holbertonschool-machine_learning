#!/usr/bin/env python3
"""
This module contains a function to update weights using
gradient descent with Dropout regularization.
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout
    regularization using gradient descent.

    Args:
        Y: one-hot numpy.ndarray (classes, m) containing correct labels
        weights: dictionary of the weights and biases
        cache: dictionary of the outputs and dropout masks of each layer
        alpha: the learning rate
        keep_prob: the probability that a node will be kept
        L: the number of layers of the network
    """
    m = Y.shape[1]
    # Starting dZ for the output layer (Softmax: A_L - Y)
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        # Calculate weight and bias gradients
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            # Backpropagate error to the previous layer
            dA_prev = np.dot(W.T, dZ)

            # 1. Apply the dropout mask from the forward pass
            dA_prev *= cache['D' + str(i - 1)]

            # 2. Scale the gradient (Inverted Dropout)
            dA_prev /= keep_prob

            # 3. Calculate dZ for the tanh hidden layer: dA * g'(Z)
            # Derivative of tanh(Z) is (1 - A^2)
            dZ = dA_prev * (1 - np.power(A_prev, 2))

        # Update weights and biases in place
        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db

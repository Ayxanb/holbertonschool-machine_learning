#!/usr/bin/env python3
"""
This module contains a function to calculate the cost of a 
neural network with L2 regularization included.
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: the cost of the network without L2 regularization
        lambtha: the L2 regularization parameter
        weights: a dictionary of the weights and biases (numpy.ndarrays)
                 of the neural network
        L: the number of layers in the neural network
        m: the number of data points used

    Returns:
        The cost of the network accounting for L2 regularization.
    """
    # Initialize the sum of squared weights
    squared_weights_sum = 0

    # Iterate through the layers to sum the Frobenius norm of each weight matrix
    for i in range(1, L + 1):
        key = 'W' + str(i)
        # We only regularize weights (W), not biases (b)
        if key in weights:
            squared_weights_sum += np.sum(np.square(weights[key]))

    # L2 Regularization formula: Cost + (lambda / 2m) * sum(weights^2)
    l2_cost = cost + (lambtha / (2 * m)) * squared_weights_sum

    return l2_cost

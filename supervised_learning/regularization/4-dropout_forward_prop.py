#!/usr/bin/env python3
"""
This module contains a function to conduct forward propagation
using Dropout in a neural network.
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Inverted Dropout.

    Args:
        X: numpy.ndarray (nx, m) containing the input data
        weights: dictionary of the weights and biases
        L: the number of layers in the network
        keep_prob: the probability that a node will be kept

    Returns:
        A dictionary containing the outputs of each layer and the
        dropout mask used on each layer.
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        # Linear Forward Step: Z = W * A_prev + b
        Z = np.dot(W, A_prev) + b

        if i == L:
            # Output Layer: Softmax activation
            t = np.exp(Z)
            cache['A' + str(i)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            # Hidden Layers: Tanh activation
            A = np.tanh(Z)

            # Create Dropout Mask
            # Random values between 0 and 1; True if < keep_prob
            mask = (np.random.rand(A.shape[0],
                    A.shape[1]) < keep_prob).astype(int)

            # Apply mask and scale (Inverted Dropout)
            A *= mask
            A /= keep_prob

            cache['D' + str(i)] = mask
            cache['A' + str(i)] = A

    return cache

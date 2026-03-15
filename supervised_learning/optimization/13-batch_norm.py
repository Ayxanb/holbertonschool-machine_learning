#!/usr/bin/env python3
"""
This module contains a function to perform batch normalization using numpy.
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network 
    using batch normalization.

    Args:
        Z (numpy.ndarray): Matrix of shape (m, n) to be normalized.
        gamma (numpy.ndarray): Scales used for batch norm, shape (1, n).
        beta (numpy.ndarray): Offsets used for batch norm, shape (1, n).
        epsilon (float): Small number to avoid division by zero.

    Returns:
        numpy.ndarray: The normalized Z matrix.
    """
    # 1. Calculate the mean of each feature (column)
    # m is the number of data points, so we average across axis 0
    mu = np.mean(Z, axis=0)

    # 2. Calculate the variance of each feature
    var = np.var(Z, axis=0)

    # 3. Normalize the data
    # Standardize to mean 0 and variance 1
    Z_centered = Z - mu
    Z_hat = Z_centered / np.sqrt(var + epsilon)

    # 4. Scale and shift (the Learnable Parameters)
    # This transforms the distribution to have mean beta and variance gamma^2
    out = gamma * Z_hat + beta

    return out

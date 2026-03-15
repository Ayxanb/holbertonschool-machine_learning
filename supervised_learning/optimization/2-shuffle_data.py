#!/usr/bin/env python3
"""
This module contains a function to shuffle two matrices in synchronization.
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Args:
        X (numpy.ndarray): Matrix of shape (m, nx) to shuffle.
        Y (numpy.ndarray): Matrix of shape (m, ny) to shuffle.

    Returns:
        tuple: (X_shuffled, Y_shuffled)
    """
    # Get the number of data points
    m = X.shape[0]

    # Create a permutation of indices from 0 to m-1
    permutation = np.random.permutation(m)

    # Use the same permutation to reorder both X and Y
    return X[permutation], Y[permutation]

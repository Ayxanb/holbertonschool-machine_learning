#!/usr/bin/env python3
"""
This module contains a function to calculate normalization constants.
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Args:
        X (numpy.ndarray): The matrix of shape (m, nx) to normalize.
            m is the number of data points, nx is the number of features.

    Returns:
        tuple: (mean, std) where:
            mean is a numpy.ndarray containing the mean of each feature.
            std is a numpy.ndarray containing the standard deviation
            of each feature.
    """
    # Calculate mean along axis 0 (columns)
    mean = np.mean(X, axis=0)

    # Calculate standard deviation along axis 0 (columns)
    std = np.std(X, axis=0)

    return mean, std

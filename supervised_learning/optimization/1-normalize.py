#!/usr/bin/env python3
"""
This module contains a function to normalize a matrix.
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix using mean and standard deviation.

    Args:
        X (numpy.ndarray): The matrix of shape (d, nx) to normalize.
        m (numpy.ndarray): The mean of all features of X, shape (nx,).
        s (numpy.ndarray): The standard deviation of all features of X,
            shape (nx,).

    Returns:
        numpy.ndarray: The normalized X matrix.
    """
    # Standardization formula: (X - mean) / std_dev
    return (X - m) / s

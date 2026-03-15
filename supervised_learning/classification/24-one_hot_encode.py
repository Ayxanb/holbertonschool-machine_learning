#!/usr/bin/env python3
import numpy as np

"""
This module provides utility functions for data preprocessing in machine learning.
Specifically, it contains a function to convert numeric label vectors into 
one-hot matrices for multi-class classification tasks.
"""


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Args:
        Y: numpy.ndarray with shape (m,) containing numeric class labels.
        classes: The maximum number of classes found in Y.

    Returns:
        A one-hot encoding of Y with shape (classes, m), or None on failure.
    """

    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None

    try:
        # Create an identity matrix of size (classes, classes)
        # Use Y to index into it, selecting the appropriate rows
        # Transpose (.T) to get shape (classes, m)
        one_hot = np.eye(classes)[Y].T
        return one_hot
    except Exception:
        # This handles cases where Y contains indices >= classes
        return None

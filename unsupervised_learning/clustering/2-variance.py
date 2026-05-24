#!/usr/bin/env python3
'''
This module contains `variance` function
'''

import numpy as np


def variance(X, C):
    """
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        C: numpy.ndarray of shape (k, d) containing the centroid means.

    Returns:
        var: The total intra-cluster variance, or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    try:
        diff = X[:, None, :] - C[None, :, :]
        sq_distances = np.sum(diff**2, axis=2)
        min_sq_distances = np.min(sq_distances, axis=1)
        total_variance = np.sum(min_sq_distances)

        return total_variance
    except Exception:
        return None

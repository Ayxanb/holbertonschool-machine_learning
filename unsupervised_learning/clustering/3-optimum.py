#!/usr/bin/env python3

'''
This module contains `optimum_k` function
that tests for the optimum number of clusters by variance
'''

import numpy as np

kmeans = __import__("1-kmeans").kmeans
variance = __import__("2-variance").variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        kmin: Positive integer, minimum number of clusters to check.
        kmax: Positive integer, maximum number of clusters to check.
        iterations: Positive integer, max iterations for K-means.

    Returns:
        results, d_vars:
            results: list containing K-means outputs for each cluster size.
            d_vars: list containing the variance differences from kmin.
            Or None, None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    variances = []

    # Loop 1: Run K-means and calculate variance for each k
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None

        results.append((C, clss))

        var = variance(X, C)
        if var is None:
            return None, None
        variances.append(var)

    d_vars = []
    base_variance = variances[0]

    # Loop 2: Calculate the difference in variance relative to kmin
    for var in variances:
        d_vars.append(base_variance - var)

    return results, d_vars

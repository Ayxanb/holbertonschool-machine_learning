#!/usr/bin/env python3
"""This module contains the `initialize` function for GMM.

It sets up priors, cluster means, and covariance matrices.
"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        k: Positive integer containing the number of clusters.

    Returns:
        pi, m, S:
            pi: numpy.ndarray of shape (k,) containing cluster priors.
            m: numpy.ndarray of shape (k, d) containing centroid means.
            S: numpy.ndarray of shape (k, d, d) containing covariances.
            Or None, None, None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    # Step 1: Run K-means to extract cluster means (m)
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    d = X.shape[1]

    # Step 2: Initialize priors (pi) evenly across clusters
    pi = np.full((k,), 1.0 / k)

    # Step 3: Initialize covariance matrices (S) as identity matrices
    # np.tile replicates the (d, d) identity matrix k times along axis 0
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S

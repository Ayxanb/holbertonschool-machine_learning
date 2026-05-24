#!/usr/bin/env python3

'''
This module implements `kmeans` function
'''

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    try:
        low = np.min(X, axis=0)
        high = np.max(X, axis=0)
        centroids = np.random.uniform(low, high, size=(k, X.shape[1]))
        return centroids
    except Exception:
        return None


def kmeans(X, k, iterations=1000):
    """Performs K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: Positive integer containing the number of clusters.
        iterations: Positive integer containing max iterations.

    Returns:
        C, clss: Centroids (k, d) and cluster assignments (n,),
                 or None, None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    C = initialize(X, k)
    if C is None:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    # Loop 1: Main optimization loop
    for _ in range(iterations):
        C_old = C.copy()

        # Compute Euclidean distances using broadcasting: shape (n, k)
        distances = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)

        # FIX: Find closest centroid across columns (axis 1), yielding (n,)
        clss = np.argmin(distances, axis=1)

        # Loop 2: Iterating through each cluster to update its mean
        for j in range(k):
            points_in_cluster = X[clss == j]

            if len(points_in_cluster) > 0:
                C[j] = np.mean(points_in_cluster, axis=0)
            else:
                # Reinitialize empty cluster centroid
                C[j] = np.random.uniform(low, high, size=(1, X.shape[1]))

        if np.all(C == C_old):
            break

    return C, clss

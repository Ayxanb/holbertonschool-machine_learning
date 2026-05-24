#!/usr/bin/env python3

'''
This module contains `initialize` function
that initializes K-means.
'''
import numpy as np


def initialize(X, k):
    '''
    X: a numpy.ndarray of shape (n, d)
        containing the dataset that will be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point

    k: a positive integer containing the number of clusters
    '''

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    try:
        # Get the min and max values along each dimension (axis 0)
        low = np.min(X, axis=0)
        high = np.max(X, axis=0)

        # Generate centroids of shape (k, d) using exactly one uniform call
        centroids = np.random.uniform(low, high, size=(k, X.shape[1]))

        return centroids
    except Exception:
        return None

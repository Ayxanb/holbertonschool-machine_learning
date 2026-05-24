#!/usr/bin/env python3
"""This module contains the `pdf` function.

It computes the probability density function of a multivariate Gaussian.
"""
import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function of a Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data points.
        m: numpy.ndarray of shape (d,) containing the mean of the distribution.
        S: numpy.ndarray of shape (d, d) containing the covariance matrix.

    Returns:
        P: numpy.ndarray of shape (n,) containing the PDF values,
           or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape

    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    try:
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)
        norm_factor = 1.0 / np.sqrt(((2 * np.pi) ** d) * det_S)
        dev = X - m

        mahalanobis = np.sum(np.dot(dev, inv_S) * dev, axis=1)

        P = norm_factor * np.exp(-0.5 * mahalanobis)
        P = np.maximum(P, 1e-300)

        return P
    except Exception:
        return None

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
        # Step 1: Calculate the determinant and inverse of covariance S
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)

        # Step 2: Calculate the normalization coefficient
        # 1 / sqrt((2 * pi)^d * det(S))
        norm_factor = 1.0 / np.sqrt(((2 * np.pi) ** d) * det_S)

        # Step 3: Compute deviation from mean (x - m) for all points -> (n, d)
        dev = X - m

        # Step 4: Calculate the Mahalanobis distance component
        # Vectorized equivalent of (x - m)^T * S^-1 * (x - m) without loops:
        # np.dot(dev, inv_S) gives (n, d) where each row is (x - m)^T * S^-1.
        # Multiplying element-wise by dev (* dev) and summing along axis 1
        # yields the final scalar product for each of the n data points.
        mahalanobis = np.sum(np.dot(dev, inv_S) * dev, axis=1)

        # Step 5: Compute probabilities and apply the floor limit
        P = norm_factor * np.exp(-0.5 * mahalanobis)
        P = np.maximum(P, 1e-300)

        return P
    except Exception:
        return None

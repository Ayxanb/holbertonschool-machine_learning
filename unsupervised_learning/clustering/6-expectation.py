#!/usr/bin/env python3
"""This module contains the `expectation` function.

It computes posterior probabilities and log likelihood for a GMM.
"""
import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        pi: numpy.ndarray of shape (k,) containing cluster priors.
        m: numpy.ndarray of shape (k, d) containing centroid means.
        S: numpy.ndarray of shape (k, d, d) containing covariances.

    Returns:
        g, l:
            g: numpy.ndarray of shape (k, n) containing posteriors.
            l: scalar value representing total log likelihood.
            Or None, None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None

    # Check that priors sum up close to 1
    if not np.isclose(np.sum(pi), 1.0):
        return None, None

    try:
        # Pre-allocate matrix matching geometry (k, n)
        probs = np.zeros((k, n))

        # Single allowed loop to populate cluster probabilities
        for i in range(k):
            cluster_pdf = pdf(X, m[i], S[i])
            if cluster_pdf is None:
                return None, None
            probs[i] = pi[i] * cluster_pdf

        # Sum individual columns to get marginal probabilities per point
        marginal_densities = np.sum(probs, axis=0)

        # Log likelihood calculation
        lg = np.sum(np.log(marginal_densities))

        # Normalize to find posterior distribution matrix
        g = probs / marginal_densities

        return g, lg
    except Exception:
        return None, None

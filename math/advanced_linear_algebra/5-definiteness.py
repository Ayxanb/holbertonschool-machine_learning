#!/usr/bin/env python3
'''
This module contains `definiteness` function
'''

import numpy as np


def definiteness(matrix):
    """
    Determines the definiteness of a square matrix.
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    if np.all(eigenvalues > 0):
        return "Positive definite"

    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"

    elif np.all(eigenvalues < 0):
        return "Negative definite"

    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"

    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"

    else:
        return None

#!/usr/bin/env python3
"""Simple policy function."""

import numpy as np


def policy(matrix, weight):
    """Compute the policy probabilities for a given state and weight matrix.

    Args:
        matrix: State/observation matrix.
        weight: Weight matrix.

    Returns:
        A vector of action probabilities.
    """
    z = np.matmul(matrix, weight)
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / np.sum(exp)


#!/usr/bin/env python3
"""
This module provides utility functions for data preprocessing in ML.

Contains functions to transform label formats, specifically converting
numeric label vectors into one-hot encoded matrices for use in
multi-class classification neural networks.
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    The conversion uses NumPy's fancy indexing on an identity matrix.
    Each element in Y serves as an index to select a row from the
    identity matrix, which is then transposed to align with the
    (classes, m) requirement.

    Args:
        Y (numpy.ndarray): A numeric label vector with shape (m,).
        classes (int): The total number of classes.

    Returns:
        numpy.ndarray: A one-hot encoding of Y with shape (classes, m).
        None: If Y is not a numpy.ndarray, classes is not an int,
              or if an index in Y exceeds the class range.
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None

    try:
        return np.eye(classes)[Y].T
    except Exception:
        return None

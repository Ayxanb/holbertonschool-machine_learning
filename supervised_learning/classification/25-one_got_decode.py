#!/usr/bin/env python3
"""
This module provides utility functions for data preprocessing in ML.

Contains functions to transform label formats, specifically converting
between numeric label vectors and one-hot encoded matrices.
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Args:
        Y (numpy.ndarray): A numeric label vector with shape (m,).
        classes (int): The total number of classes.

    Returns:
        numpy.ndarray: A one-hot encoding of Y with shape (classes, m).
        None: On failure.
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None

    try:
        return np.eye(classes)[Y].T
    except Exception:
        return None


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    Args:
        one_hot (numpy.ndarray): A one-hot encoded matrix with
            shape (classes, m).

    Returns:
        numpy.ndarray: A vector with shape (m,) containing the numeric
            labels for each example.
        None: On failure.
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    # axis=0 identifies the index of the '1' across the class rows
    return np.argmax(one_hot, axis=0)

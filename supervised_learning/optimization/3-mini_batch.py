#!/usr/bin/env python3
"""
This module contains a function to create mini-batches for training.
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches for mini-batch gradient descent.

    Args:
        X (numpy.ndarray): Input data of shape (m, nx).
        Y (numpy.ndarray): Labels of shape (m, ny).
        batch_size (int): Number of data points in each batch.

    Returns:
        list: A list of tuples (X_batch, Y_batch).
    """
    # 1. Shuffle the data so batches aren't biased by the original order
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    m = X.shape[0]
    mini_batches = []

    # 2. Iterate through the data in chunks of batch_size
    for i in range(0, m, batch_size):
        # Slice X and Y from current index to index + batch_size
        # Python's slicing handles the 'end of list' automatically
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]

        mini_batches.append((X_batch, Y_batch))

    return mini_batches

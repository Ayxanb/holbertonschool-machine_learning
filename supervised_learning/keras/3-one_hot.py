#!/usr/bin/env python3

"""
This module contains a function to convert label vectors
into one-hot matrices using the Keras/TensorFlow library.
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
        labels: a array-like object containing numerical labels
        classes: the total number of classes. If None, it will be
                 inferred from the maximum value in labels.

    Returns:
        The one-hot matrix as a numpy array.
    """

    # K.utils.to_categorical is the standard way to handle
    # one-hot encoding in Keras.
    one_hot_matrix = K.utils.to_categorical(labels, num_classes=classes)

    return one_hot_matrix

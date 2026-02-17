#!/usr/bin/env python3

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot encoded labels and logits.
    """

    # One-hot labels  ->  1D array of class indices
    # (m, classes) -> (m,)
    actual = np.argmax(labels, axis=1)
    predicted = np.argmax(logits, axis=1)

    # Number of classes from the input shape
    num_classes = labels.shape[1]

    # Empty confusion matrix of zeros
    confusion = np.zeros((num_classes, num_classes))

    # Each 'a' is the row (Actual), each 'p' is the column (Predicted)
    for a, p in zip(actual, predicted):
        confusion[a, p] += 1

    return confusion

#!/usr/bin/env python3

"""
This module contains a function to train a Keras model
using mini-batch gradient descent.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes)
                containing the labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data
        verbose: boolean determining if output should be printed
        shuffle: boolean determining whether to shuffle batches every epoch

    Returns:
        The History object generated after training the model.
    """

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

    return history

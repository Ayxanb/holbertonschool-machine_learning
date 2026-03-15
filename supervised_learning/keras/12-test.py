#!/usr/bin/env python3
"""
This module contains a function to test a Keras model
on a specific dataset.
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network and returns its performance.

    Args:
        network: the network model to test
        data: the input data to test the model with
        labels: the correct one-hot labels of data
        verbose: boolean determining if output should be printed

    Returns:
        A list containing the loss and accuracy of the model, respectively.
    """
    # evaluate returns a list: [loss, accuracy, ...]
    results = network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )

    return results

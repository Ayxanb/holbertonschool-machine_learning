#!/usr/bin/env python3
"""
This module contains a function to make predictions
using a trained Keras model.
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    Args:
        network: the network model to make the prediction with
        data: the input data to make the prediction with
        verbose: boolean determining if output should be printed
                 during the prediction process

    Returns:
        The prediction for the data (numpy.ndarray).
    """
    # The predict method returns the output of the last layer
    prediction = network.predict(
        x=data,
        verbose=verbose
    )

    return prediction

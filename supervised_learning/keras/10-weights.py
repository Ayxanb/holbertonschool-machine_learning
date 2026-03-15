#!/usr/bin/env python3
"""
This module contains functions to save and load only
the weights of a Keras model.
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights to a file.

    Args:
        network: the model whose weights should be saved
        filename: the path of the file that the weights should be saved to
        save_format: the format in which the weights should be saved
                     (e.g., 'keras', 'h5', or 'tf')

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Loads a model's weights from a file.

    Args:
        network: the model to which the weights should be loaded
        filename: the path of the file that the weights should be loaded from

    Returns:
        None
    """
    network.load_weights(filename)
    return None

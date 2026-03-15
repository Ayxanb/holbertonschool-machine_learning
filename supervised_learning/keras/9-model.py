#!/usr/bin/env python3
"""
This module contains functions to save and load Keras models.
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model to a file.

    Args:
        network: the model to save
        filename: the path of the file where the model should be saved

    Returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model from a file.

    Args:
        filename: the path of the file from which the model should be loaded

    Returns:
        The loaded Keras model.
    """
    return K.models.load_model(filename)

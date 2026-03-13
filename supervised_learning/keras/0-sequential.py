#!/usr/bin/env python3
"""
This module provides a function to build a customizable deep neural network
using the Keras Sequential API, incorporating L2 regularization and Dropout.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

    Args:
        nx (int): The number of input features to the network.
        layers (list): A list of integers representing the number of nodes
            in each layer of the network.
        activations (list): A list of strings representing the activation
            functions to be used for each layer.
        lambtha (float): The L2 regularization parameter (weight decay).
        keep_prob (float): The probability that a node will be kept
            during Dropout.

    Returns:
        keras.Model: A Keras Sequential model instance configured with the
        specified architecture, regularization, and dropout settings.
    """

    regularizer = K.regularizers.l2(lambtha)
    model = K.Sequential()

    for i in range(len(layers)):
        """
        Iterate through the layers list to construct the network architecture.
        The first layer is specifically assigned the input_shape derived from nx.
        """
        if i == 0:
            model.add(K.layers.Dense(
                layers[i],
                input_shape=(nx,),
                activation=activations[i],
                kernel_regularizer=regularizer
            ))
        else:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer
            ))

        if i < len(layers) - 1:
            """
            Apply Dropout to all hidden layers. The dropout rate is
            calculated as (1 - keep_prob) to match Keras specifications.
            """
            model.add(K.layers.Dropout(1 - keep_prob))

    return model

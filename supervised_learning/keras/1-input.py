#!/usr/bin/env python3

'''
This module provides a function to build a customizable deep neural network
using the Keras Functional API, incorporating L2 regularization and Dropout.
'''

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library using the Functional API.

    Args:
        nx: number of input features
        layers: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns:
        The compiled Keras model.
    """

    inputs = K.Input(shape=(nx,))
    regularizer = K.regularizers.l2(lambtha)

    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer
        )(x)

        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    return K.Model(inputs=inputs, outputs=x)

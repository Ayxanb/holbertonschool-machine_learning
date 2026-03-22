#!/usr/bin/env python3
"""
This module contains a function to create a Keras layer
with L2 regularization.
"""
import tensorflow.keras as K


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow that
    includes L2 regularization.

    Args:
        prev: a tensor containing the output of the previous layer
        n: the number of nodes the new layer should contain
        activation: the activation function that should be used on the layer
        lambtha: the L2 regularization parameter

    Returns:
        The output tensor of the new layer.
    """
    # Initialize the L2 regularizer with the given lambtha
    regularizer = K.regularizers.L2(lambtha)

    # Define the Dense layer with regularization and activation
    layer = K.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizer,
        kernel_initializer=K.initializers.VarianceScaling(
            scale=2.0, mode='fan_avg', distribution='normal'
        )
    )

    # Return the output of the layer applied to the previous tensor
    return layer(prev)

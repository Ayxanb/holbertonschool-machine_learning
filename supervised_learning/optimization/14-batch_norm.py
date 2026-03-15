#!/usr/bin/env python3
"""
Sets up a Batch Normalization layer in tensorFlow.
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.

    Args:
        prev: The activated output of the previous layer.
        n: The number of nodes in the layer to be created.
        activation: The activation function to be used on the output.

    Returns:
        A tensor of the activated output for the layer.
    """
    # 1. Base Dense Layer (Linear Transformation)
    # We use VarianceScaling(mode='fan_avg') as requested.
    # Note: use_bias=False is common when using Batch Norm because
    # the 'beta' parameter in BN acts as the bias.
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )
    Z = dense_layer(prev)

    # 2. Batch Normalization Layer
    # gamma_initializer='ones' and beta_initializer='zeros' are defaults,
    # but we specify them here for clarity.
    batch_norm = tf.keras.layers.BatchNormalization(
        beta_initializer='zeros',
        gamma_initializer='ones',
        epsilon=1e-7
    )
    Z_norm = batch_norm(Z)

    # 3. Apply Activation
    # If activation is None, it returns the normalized linear output.
    if activation is None:
        return Z_norm

    return activation(Z_norm)

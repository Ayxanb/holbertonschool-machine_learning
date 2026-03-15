#!/usr/bin/env python3
"""
Sets up a Batch Normalization layer in tensorflow.
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
    # 1. Base Dense Layer
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )
    Z = dense(prev)

    # 2. Batch Normalization Layer
    bn_layer = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        beta_initializer='zeros',
        gamma_initializer='ones'
    )

    # Pass training=True to force the layer to use the
    # current batch statistics (mean/variance)
    Z_norm = bn_layer(Z, training=True)

    # 3. Apply Activation
    if activation is None:
        return Z_norm

    return activation(Z_norm)

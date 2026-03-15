#!/usr/bin/env python3
"""
Sets up a Batch Normalization layer in TensorFlow 2.x.
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
    # Initialize with VarianceScaling(mode='fan_avg')
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    
    # We create the layer and IMMEDIATELY call it on 'prev' to get Z
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )
    Z = dense(prev)

    # 2. Batch Normalization Layer
    # Create the layer object
    bn_layer = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        beta_initializer='zeros',
        gamma_initializer='ones'
    )
    
    # IMPORTANT: Call the BN layer on the linear output Z
    # This applies the normalization, scaling (gamma), and shifting (beta)
    Z_norm = bn_layer(Z)

    # 3. Apply Activation
    if activation is None:
        return Z_norm
    
    return activation(Z_norm)

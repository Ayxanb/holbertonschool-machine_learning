#!/usr/bin/env python3
"""
This module contains a function to create a Keras layer
incorporating dropout regularization using the tf import.
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Args:
        prev: tensor containing the output of the previous layer
        n: the number of nodes the new layer should contain
        activation: the activation function for the new layer
        keep_prob: the probability that a node will be kept
        training: boolean indicating whether the model is in training mode

    Returns:
        The output tensor of the new layer.
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')

    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )
    x = dense_layer(prev)

    rate = 1 - keep_prob
    dropout_layer = tf.keras.layers.Dropout(rate=rate)

    return dropout_layer(x, training=training)

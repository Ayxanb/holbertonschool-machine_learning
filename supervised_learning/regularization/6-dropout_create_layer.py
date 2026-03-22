#!/usr/bin/env python3
"""
This module contains a function to create a Keras layer 
incorporating dropout regularization.
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
    # Initialize the Dense layer
    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=K.initializers.VarianceScaling(
            scale=2.0, mode='fan_avg', distribution='normal'
        )
    )
    
    # Pass the previous output through the dense layer
    x = dense_layer(prev)
    
    # Define the Dropout layer. Note: rate = 1 - keep_prob
    dropout_layer = tf.keras.layers.Dropout(rate=1 - keep_prob)
    
    # Apply dropout to the output of the dense layer
    # The 'training' argument ensures dropout is only active during training
    return dropout_layer(x, training=training)

#!/usr/bin/env python3

"""
This module contains a function to set up Adam optimization 
for a Keras model using categorical crossentropy.
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a Keras model with
    categorical crossentropy loss and accuracy metrics.

    Args:
        network: the model to optimize
        alpha: the learning rate
        beta1: the first Adam optimization parameter
        beta2: the second Adam optimization parameter

    Returns:
        None
    """

    optimizer = K.optimizers.Adam(
            learning_rate=alpha,
            beta_1=beta1,
            beta_2=beta2
    )

    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

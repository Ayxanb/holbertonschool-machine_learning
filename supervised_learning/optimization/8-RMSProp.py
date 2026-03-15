#!/usr/bin/env python3
"""
Sets up the RMSProp optimization algorithm in TensorFlow 2.x.
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow 2.x.

    Args:
        alpha: The learning rate.
        beta2: The RMSProp weight (discounting factor).
        epsilon: A small number to avoid division by zero.

    Returns:
        The optimizer object.
    """
    # In Keras, beta2 is referred to as 'rho'
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )

    return optimizer

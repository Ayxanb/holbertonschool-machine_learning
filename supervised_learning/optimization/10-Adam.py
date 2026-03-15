#!/usr/bin/env python3
"""
Sets up the Adam optimization algorithm in TensorFlow 2.x.
"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up the Adam optimization algorithm in TensorFlow 2.x.

    Args:
        alpha: The learning rate.
        beta1: The weight used for the first moment.
        beta2: The weight used for the second moment.
        epsilon: A small number to avoid division by zero.

    Returns:
        The optimizer object.
    """
    # tf.keras.optimizers.Adam implements the adaptive moment estimation.
    # It combines momentum and RMSProp logic.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )

    return optimizer

#!/usr/bin/env python3
"""
Sets up the Momentum optimization algorithm in TensorFlow 2.x.
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization 
    algorithm in TensorFlow 2.x.

    Args:
        alpha: The learning rate.
        beta1: The momentum weight.

    Returns:
        The optimizer object.
    """
    # In TF 2.x, momentum is part of the SGD (Stochastic Gradient Descent) 
    # optimizer class.
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )

    return optimizer

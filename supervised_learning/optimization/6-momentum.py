#!/usr/bin/env python3
"""
Sets up the Momentum optimization algorithm.
"""
import tensorflow.keras as K


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm.

    Args:
        alpha: The learning rate.
        beta1: The momentum weight.

    Returns:
        The optimizer object.
    """
    return K.optimizers.SGD(
            learning_rate=alpha,
            momentum=beta1
    )

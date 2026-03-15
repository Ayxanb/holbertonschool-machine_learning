#!/usr/bin/env python3
"""
Sets up inverse time learning rate decay in tensorflow.
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in TensorFlow using
    inverse time decay.

    Args:
        alpha: The original learning rate.
        decay_rate: The weight used to determine the rate of decay.
        decay_step: The number of passes before alpha is decayed further.

    Returns:
        A learning rate decay schedule (operation).
    """
    # InverseTimeDecay implements the formula:
    # lr = initial_lr / (1 + decay_rate * floor(step / decay_step))
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True  # This ensures the stepwise/staircase fashion
    )

    return lr_schedule

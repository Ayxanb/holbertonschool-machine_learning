#!/usr/bin/env python3
"""
This module contains a function for stepwise inverse time decay.
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in a stepwise fashion.

    Args:
        alpha (float): The original learning rate.
        decay_rate (float): The weight used to determine the rate of decay.
        global_step (int): Number of passes of gradient descent elapsed.
        decay_step (int): Number of passes before alpha is decayed further.

    Returns:
        float: The updated value for alpha.
    """
    # Use floor division to create the "staircase" or stepwise effect
    # The decay only updates every 'decay_step' iterations
    num_decays = global_step // decay_step

    # Calculate the inverse time decay
    alpha_new = alpha / (1 + decay_rate * num_decays)

    return alpha_new

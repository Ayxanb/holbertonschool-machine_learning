#!/usr/bin/env python3
"""
This module contains a function to update variables using Momentum.
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using Gradient Descent with Momentum.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of var.
        v (numpy.ndarray): The previous first moment of var.

    Returns:
        tuple: (updated_variable, new_moment)
    """
    # Calculate the new moment (velocity)
    # v is weighted by beta1, and the current gradient is weighted by (1-beta1)
    new_moment = beta1 * v + (1 - beta1) * grad

    # Update the variable using the new moment instead of just the raw gradient
    updated_var = var - alpha * new_moment

    return updated_var, new_moment

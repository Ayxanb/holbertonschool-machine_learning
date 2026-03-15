#!/usr/bin/env python3
"""
This module contains a function to update variables using RMSProp.
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight (discounting factor).
        epsilon (float): Small number to avoid division by zero.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of var.
        s (numpy.ndarray): The previous second moment of var.

    Returns:
        tuple: (updated_variable, new_moment)
    """
    # 1. Update the second moment (running average of squared gradients)
    new_moment = beta2 * s + (1 - beta2) * (grad ** 2)

    # 2. Update the variable
    # We divide the gradient by the square root of the second moment.
    # This scales down large gradients and scales up small gradients.
    updated_var = var - alpha * (grad / (np.sqrt(new_moment) + epsilon))

    return updated_var, new_moment

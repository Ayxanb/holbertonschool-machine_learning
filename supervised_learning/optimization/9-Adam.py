#!/usr/bin/env python3
"""
This module contains a function to update variables using Adam.
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The weight for the first moment.
        beta2 (float): The weight for the second moment.
        epsilon (float): Small number to avoid division by zero.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of var.
        v (numpy.ndarray): The previous first moment of var.
        s (numpy.ndarray): The previous second moment of var.
        t (int): The time step used for bias correction.

    Returns:
        tuple: (updated_variable, new_first_moment, new_second_moment)
    """
    # 1. Update the moments
    v_new = beta1 * v + (1 - beta1) * grad
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # 2. Apply bias correction
    # We use t (the current time step) to scale the moments
    v_corrected = v_new / (1 - (beta1 ** t))
    s_corrected = s_new / (1 - (beta2 ** t))

    # 3. Update the variable in place
    var -= alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return var, v_new, s_new

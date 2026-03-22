#!/usr/bin/env python3
"""
This module contains a function to calculate the total cost
of a Keras model including L2 regularization.
"""
import tensorflow.keras as K


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: a tensor containing the cost of the network
              without L2 regularization
        model: a Keras model that includes layers with L2 regularization

    Returns:
        A tensor containing the total cost accounting for L2 regularization.
    """
    # Keras models store regularization losses in the 'losses' attribute
    # We add the base cost to the sum of these regularization penalties
    total_cost = cost + model.losses
    
    return total_cost

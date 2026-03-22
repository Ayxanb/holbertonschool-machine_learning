#!/usr/bin/env python3
"""
This module contains a function to determine if gradient
descent should be stopped early.
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early.

    Args:
        cost: the current validation cost of the neural network
        opt_cost: the lowest recorded validation cost of the network
        threshold: the threshold used for early stopping
        patience: the patience count used for early stopping
        count: the count of how long the threshold has not been met

    Returns:
        A boolean (stop) and the updated count (count).
    """
    # Check if the improvement is greater than the threshold
    # Formula: opt_cost - cost > threshold
    if opt_cost - cost > threshold:
        # Reset count if significant improvement is made
        count = 0
    else:
        # Increment count if improvement is stagnant or cost rises
        count += 1

    # Stop if the count reaches or exceeds patience
    stop = count >= patience

    return stop, count

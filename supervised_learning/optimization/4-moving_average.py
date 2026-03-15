#!/usr/bin/env python3
"""
This module contains a function to calculate the weighted moving average.
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set with bias correction.

    Args:
        data (list): The list of data to calculate the moving average of.
        beta (float): The weight used for the moving average.

    Returns:
        list: A list containing the moving averages of data.
    """
    v = 0
    moving_averages = []

    for t, theta in enumerate(data, 1):
        # Update the exponentially weighted moving average
        v = beta * v + (1 - beta) * theta

        # Apply bias correction
        # t is the current time step (starting from 1)
        v_corrected = v / (1 - (beta ** t))

        moving_averages.append(v_corrected)

    return moving_averages

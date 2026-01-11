#!/usr/bin/env python3
"""
This module provides a function to create a scatter plot showing
the relationship between men's height and weight.
"""

import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Generates and displays a scatter plot of men's height versus weight.

    The data is randomly generated using a multivariate normal distribution.
    Heights are measured in inches, weights in pounds.
    """

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]

    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180

    plt.figure(figsize=(6.4, 4.8))
    plt.title("Men's Height vs Weight")
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")
    plt.scatter(x, y, color='magenta')
    plt.show()

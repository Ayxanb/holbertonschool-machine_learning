#!/usr/bin/env python3
"""
This module provides a function to plot the exponential decay of Carbon-14
using a logarithmic scale.
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots the fraction of Carbon-14 remaining over time.

    The x-axis represents time in years, and the y-axis represents
    the fraction remaining on a logarithmic scale.
    """

    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y)
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')
    plt.xlim(0, 28650)
    plt.yscale('log')
    plt.ylim(bottom=y[-1])
    plt.show()

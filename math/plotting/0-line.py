#!/usr/bin/env python3

'''
This module provides a `line` function which plots x^3
'''

import numpy as np
import matplotlib.pyplot as plt


def line():
    '''
    plots x^3
    '''
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, color='red')

    plt.show()

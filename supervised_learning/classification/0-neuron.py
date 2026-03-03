#!/usr/bin/env python3

'''
This module contains `Neuron` class implementation.
'''

import numpy as np


class Neuron:
    '''
    Defines a single neuron performing binary classification.
    '''

    def __init__(self, nx):
        '''
        Initializes the neuron.

        Args:
            nx (int): The number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        '''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')

        elif nx < 1:
            raise ValueError('nx must be a positive integer')

        self.nx = nx
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0

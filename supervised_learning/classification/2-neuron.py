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

        # Private attributes
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Args:
            X: numpy.ndarray with shape (nx, m)

        Returns the private attribute __A
        """

        # Linear combination: Z = WX + b
        z = self.__W @ X + self.__b

        # Sigmoid activation function
        self.__A = 1 / (1 + np.exp(-z))

        return self.__A

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

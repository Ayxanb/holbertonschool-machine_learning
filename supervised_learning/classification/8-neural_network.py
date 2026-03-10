#!/usr/bin/env python3
"""
This module defines a neural network with one hidden layer.
The network is designed for binary classification tasks using
a hidden layer of a specified size.
"""
import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer performing
    binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Initializes the NeuralNetwork class.

        Args:
            nx (int): The number of input features.
            nodes (int): The number of nodes in the hidden layer.

        Raises:
            TypeError: If nx or nodes are not integers.
            ValueError: If nx or nodes are less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        The weights vector for the hidden layer.
        """
        return self.__W1

    @property
    def b1(self):
        """
        The bias for the hidden layer.
        """
        return self.__b1

    @property
    def A1(self):
        """
        The activated output for the hidden layer.
        """
        return self.__A1

    @property
    def W2(self):
        """
        The weights vector for the output neuron.
        """
        return self.__W2

    @property
    def b2(self):
        """
        The bias for the output neuron.
        """
        return self.__b2

    @property
    def A2(self):
        """
        The activated output for the output neuron (prediction).
        """
        return self.__A2

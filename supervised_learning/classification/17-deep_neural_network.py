#!/usr/bin/env python3
"""
This module defines a deep neural network with multiple hidden layers.
It uses private attributes and He initialization for deep learning.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Class constructor for the Deep Neural Network.

        Args:
            nx (int): Number of input features.
            layers (list): Number of nodes in each layer of the network.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If layers is not a list.
            TypeError: If layers elements are not positive integers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            # Determine the input size for the current layer
            # i=0 takes input nx, otherwise it takes the previous layer size
            prev_size = nx if i == 0 else layers[i - 1]

            # He et al. initialization
            # W = random * sqrt(2 / input_size)
            self.__weights[f"W{i + 1}"] = (
                np.random.randn(layers[i], prev_size) *
                np.sqrt(2 / prev_size)
            )

            # Biases initialized to zeros
            self.__weights[f"b{i + 1}"] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the intermediary values dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights and biases dictionary."""
        return self.__weights

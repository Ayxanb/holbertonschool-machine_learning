#!/usr/bin/env python3
"""
This module defines a deep neural network with multiple hidden layers.
It initializes weights and biases for each layer using the He et al.
method for binary classification.
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
            TypeError: If layers is not a list of positive integers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            # Determine the size of the previous layer
            # layer 0 uses nx (the input features)
            prev_nodes = nx if i == 0 else layers[i - 1]

            # He et al. initialization: W = random * sqrt(2 / prev_nodes)
            self.weights[f"W{i + 1}"] = (
                np.random.randn(layers[i], prev_nodes) *
                np.sqrt(2 / prev_nodes)
            )

            # Biases initialized to zeros
            self.weights[f"b{i + 1}"] = np.zeros((layers[i], 1))

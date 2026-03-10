#!/usr/bin/env python3
import numpy as np

"""
This module constains `NeuralNetwork` class definition.
"""


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer performing binary
    classification. This class encapsulates the weights and biases
    required to transform input features into a binary prediction.
    """

    def __init__(self, nx, nodes):
        """
        Initializes the neural network.

        Args:
            nx (int): The number of input features.
            nodes (int): The number of neurons in the hidden layer.

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

        # Allows each hidden node to process every input feature.
        self.__W1 = np.random.normal(size=(nodes, nx))

        # Column vector added to the hidden layer's linear product.
        self.__b1 = np.zeros((nodes, 1))

        # __A1: Activated output for hidden layer.
        self.__A1 = 0

        # Allows the output node to process every hidden layer output.
        self.__W2 = np.random.normal(size=(1, nodes))

        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Weights vector for the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Bias for the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Activated output for the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Weights vector for the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Bias for the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Activated output for the output neuron (prediction)."""
        return self.__A2

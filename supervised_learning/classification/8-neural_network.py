#!/usr/bin/env python3
import numpy as np

"""
This module contains the `NeuralNetwork` class.
"""

    
class NeuralNetwork:
    """
    Defines a neural network with one hidden layer
    performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        Initializes the neural network

        Args:
            nx: number of input features
            nodes: number of nodes in the hidden layer
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")

        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer: W1 shape is (nodes, nx), b1 shape is (nodes, 1)
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        # Output neuron: W2 shape is (1, nodes), b2 is a scalar (or 1,1)
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0

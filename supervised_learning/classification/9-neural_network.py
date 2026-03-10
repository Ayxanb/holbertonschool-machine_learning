#!/usr/bin/env python3
import numpy as np

"""
This module contains definition of `NeuralNetwork` class.
"""


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer performing binary
    classification.

    Attributes:
        W1 (numpy.ndarray): The weights vector for the hidden layer.
        b1 (numpy.ndarray): The bias for the hidden layer.
        A1 (float): The activated output for the hidden layer.
        W2 (numpy.ndarray): The weights vector for the output neuron.
        b2 (int): The bias for the output neuron.
        A2 (float): The activated output for the output neuron.
    """

    def __init__(self, nx, nodes):
        """
        Class constructor to initialize the neural network.

        Args:
            nx (int): The number of input features to the neuron.
            nodes (int): The number of nodes found in the hidden layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If nodes is not an integer.
            ValueError: If nodes is less than 1.
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
        Getter for the weights vector of the hidden layer.
        Returns:
            numpy.ndarray: Random normal distribution of shape (nodes, nx).
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter for the bias of the hidden layer.
        Returns:
            numpy.ndarray: Zeros of shape (nodes, 1).
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter for the activated output of the hidden layer.
        Returns:
            The current activation value (initially 0).
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter for the weights vector of the output neuron.
        Returns:
            numpy.ndarray: Random normal distribution of shape (1, nodes).
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter for the bias of the output neuron.
        Returns:
            int: The current bias value (initially 0).
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter for the activated output of the output neuron.
        Returns:
            The current prediction value (initially 0).
        """
        return self.__A2

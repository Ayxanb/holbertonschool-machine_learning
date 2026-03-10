#!/usr/bin/env python3
"""
This module defines a deep neural network with multiple hidden layers.
It includes forward propagation, cost, evaluation, and gradient descent.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Initializes the Deep Neural Network.

        Args:
            nx (int): Number of input features.
            layers (list): Number of nodes in each layer.
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

            prev_size = nx if i == 0 else layers[i - 1]

            self.__weights[f"W{i + 1}"] = (
                np.random.randn(layers[i], prev_size) *
                np.sqrt(2 / prev_size)
            )
            self.__weights[f"b{i + 1}"] = np.zeros((layers[i], 1))

    def forward_prop(self, X):
        """Calculates forward propagation."""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            weights = self.__weights[f"W{i}"]
            bias = self.__weights[f"b{i}"]
            prev_activation = self.__cache[f"A{i - 1}"]
            z = np.dot(weights, prev_activation) + bias
            self.__cache[f"A{i}"] = 1 / (1 + np.exp(-z))
        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """Calculates the logistic regression cost."""
        m = Y.shape[1]
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return (1 / m) * np.sum(loss)

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions."""
        a_last, _ = self.forward_prop(X)
        cost = self.cost(Y, a_last)
        prediction = np.where(a_last >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Args:
            Y: Correct labels (1, m).
            cache: Dictionary of intermediary values.
            alpha: Learning rate.
        """
        m = Y.shape[1]
        dz = cache[f"A{self.__L}"] - Y

        for i in range(self.__L, 0, -1):
            a_prev = cache[f"A{i - 1}"]
            w_curr = self.__weights[f"W{i}"]

            dw = (1 / m) * np.dot(dz, a_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            if i > 1:
                # Derivative of sigmoid: A * (1 - A)
                # Used to calculate the gradient for the next layer back
                dz = np.dot(w_curr.T, dz) * (a_prev * (1 - a_prev))

            self.__weights[f"W{i}"] = self.__weights[f"W{i}"] - (alpha * dw)
            self.__weights[f"b{i}"] = self.__weights[f"b{i}"] - (alpha * db)

    @property
    def L(self):
        """Getter for L."""
        return self.__L

    @property
    def cache(self):
        """Getter for cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights."""
        return self.__weights

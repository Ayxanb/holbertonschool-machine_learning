#!/usr/bin/env python3
"""
This module defines a deep neural network for multiclass classification.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing multiclass classification.
    """

    def __init__(self, nx, layers):
        """Initializes the Deep Neural Network."""
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
        """
        Calculates forward propagation using Softmax for the output layer.
        """
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            weights = self.__weights[f"W{i}"]
            bias = self.__weights[f"b{i}"]
            prev_activation = self.__cache[f"A{i - 1}"]
            
            z = np.dot(weights, prev_activation) + bias
            
            if i == self.__L:
                # Softmax activation for the last layer
                t = np.exp(z)
                self.__cache[f"A{i}"] = t / np.sum(t, axis=0, keepdims=True)
            else:
                # Sigmoid activation for hidden layers
                self.__cache[f"A{i}"] = 1 / (1 + np.exp(-z))
                
        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """
        Calculates the multiclass categorical cross-entropy cost.
        """
        m = Y.shape[1]
        # Avoid division by zero/log(0) errors
        cost = -1 / m * np.sum(Y * np.log(A + 1e-8))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.
        """
        A, _ = self.forward_prop(X)
        # Convert Softmax probabilities to one-hot predictions
        prediction = np.eye(A.shape[0])[np.argmax(A, axis=0)].T
        cost = self.cost(Y, A)
        return prediction.astype(int), cost

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

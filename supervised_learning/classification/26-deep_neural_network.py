#!/usr/bin/env python3
"""
This module defines a deep neural network with multiple hidden layers.
It includes initialization, forward propagation, cost,
evaluation, gradient descent, training with visualization,
and persistence (saving/loading).
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.
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
        """Calculates one pass of gradient descent on the neural network."""
        m = Y.shape[1]
        dz = cache[f"A{self.__L}"] - Y

        for i in range(self.__L, 0, -1):
            a_prev = cache[f"A{i - 1}"]
            w_curr = self.__weights[f"W{i}"]

            dw = (1 / m) * np.dot(dz, a_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            if i > 1:
                dz = np.dot(w_curr.T, dz) * (a_prev * (1 - a_prev))

            self.__weights[f"W{i}"] = self.__weights[f"W{i}"] - (alpha * dw)
            self.__weights[f"b{i}"] = self.__weights[f"b{i}"] - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the deep neural network and optionally graphs results."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []
        for i in range(iterations + 1):
            a_last, cache = self.forward_prop(X)
            curr_cost = self.cost(Y, a_last)

            if i % step == 0 or i == iterations:
                if verbose:
                    print(f"Cost after {i} iterations: {curr_cost}")
                if graph:
                    costs.append(curr_cost)
                    iters.append(i)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(iters, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object.
        """
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)

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

#!/usr/bin/env python3
"""
This module defines a deep neural network for multiclass classification.

It includes initialization, forward propagation with Softmax,
Categorical Cross-Entropy cost, evaluation, gradient descent,
training with visualization, and persistence (saving/loading).
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing multiclass classification.
    """

    def __init__(self, nx, layers):
        """
        Initializes the Deep Neural Network.

        Args:
            nx (int): number of input features.
            layers (list): list of number of nodes in each layer.
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

            # He-et-al initialization
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
            W = self.__weights[f"W{i}"]
            b = self.__weights[f"b{i}"]
            A_prev = self.__cache[f"A{i - 1}"]
            Z = np.dot(W, A_prev) + b

            if i == self.__L:
                # Softmax activation for multiclass output
                t = np.exp(Z)
                self.__cache[f"A{i}"] = t / np.sum(t, axis=0, keepdims=True)
            else:
                # Sigmoid activation for hidden layers
                self.__cache[f"A{i}"] = 1 / (1 + np.exp(-Z))

        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """
        Calculates the multiclass categorical cross-entropy cost.
        """
        m = Y.shape[1]
        # Avoid log(0) errors with a small epsilon
        cost = -1 / m * np.sum(Y * np.log(A + 1e-8))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        # Convert probabilities to one-hot prediction format
        prediction = np.eye(A.shape[0])[np.argmax(A, axis=0)].T
        return prediction.astype(int), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent.
        """
        m = Y.shape[1]
        # dZ for Softmax/CCE is (A - Y)
        dz = cache[f"A{self.__L}"] - Y

        for i in range(self.__L, 0, -1):
            a_prev = cache[f"A{i - 1}"]
            w_curr = self.__weights[f"W{i}"]

            dw = (1 / m) * np.dot(dz, a_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            if i > 1:
                # Backprop through Sigmoid: dZ = (W.T * dZ) * g'(Z)
                dz = np.dot(w_curr.T, dz) * (a_prev * (1 - a_prev))

            self.__weights[f"W{i}"] -= (alpha * dw)
            self.__weights[f"b{i}"] -= (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network.
        """
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
            A, cache = self.forward_prop(X)
            curr_cost = self.cost(Y, A)

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
        """Saves the instance object to a file in pickle format."""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object."""
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

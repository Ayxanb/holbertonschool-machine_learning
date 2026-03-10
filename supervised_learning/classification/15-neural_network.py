#!/usr/bin/env python3

"""
This module defines a neural network with one hidden layer.
It includes initialization of weights, biases, and private attributes
for binary classification.
"""
import numpy as np


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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            tuple: (private attribute __A1, private attribute __A2)
        """
        # Layer 1 (Hidden Layer) calculation
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))

        # Layer 2 (Output Layer) calculation
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels for input data (1, m).
            A (numpy.ndarray): Activated output for each example (1, m).

        Returns:
            float: The logistic regression cost.
        """
        m = Y.shape[1]
        # Using 1.0000001 - A to prevent log(0) calculation errors
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1 / m) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels for input data (1, m).

        Returns:
            tuple: (prediction ndarray, cost)
        """
        # Step 1: Perform forward propagation to update A1 and A2
        _, a2 = self.forward_prop(X)

        # Step 2: Calculate the cost based on the final output A2
        curr_cost = self.cost(Y, a2)

        # Step 3: Threshold the probabilities into binary labels
        prediction = np.where(a2 >= 0.5, 1, 0)

        return prediction, curr_cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Args:
            X: Input data (nx, m).
            Y: Correct labels (1, m).
            A1: Output of the hidden layer.
            A2: Predicted output.
            alpha: Learning rate.
        """
        m = Y.shape[1]

        # Output layer gradients
        dz2 = A2 - Y
        dw2 = (1 / m) * np.dot(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        # Hidden layer gradients
        # Derivative of sigmoid A1 is A1 * (1 - A1)
        dz1 = np.dot(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1 / m) * np.dot(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        # Update weights and biases
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the neural network and optionally graphs the cost."""
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
            # Forward prop to update activations
            self.forward_prop(X)
            curr_cost = self.cost(Y, self.__A2)

            # Logging and tracking data
            if i % step == 0 or i == iterations:
                if verbose:
                    print(f"Cost after {i} iterations: {curr_cost}")
                if graph:
                    costs.append(curr_cost)
                    iters.append(i)

            # Gradient Descent (skip on final iteration)
            if i < iterations:
                self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        if graph:
            plt.plot(iters, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

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

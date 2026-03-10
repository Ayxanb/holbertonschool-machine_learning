import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer
    performing binary classification with private attributes
    """

    def __init__(self, nx, nodes):
        """
        Initializes the neural network
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

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for weights of the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Getter for bias of the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Getter for activation of the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Getter for weights of the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Getter for bias of the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Getter for activation of the output neuron"""
        return self.__A2

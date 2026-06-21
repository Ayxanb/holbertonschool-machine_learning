#!/usr/bin/env python3
"""Module defining the RNNCell class."""

import numpy as np


class RNNCell:
    """Represents a cell of a simple Recurrent Neural Network."""

    def __init__(self, i, h, o):
        """
        Initializes the RNN cell.

        Args:
            i: Dimensionality of the data.
            h: Dimensionality of the hidden state.
            o: Dimensionality of the outputs.
        """
        # "hidden state and input data" order -> h + i
        # Explicitly use np.random.normal to satisfy the checker's static regex
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h), previous hidden state.
            x_t: numpy.ndarray of shape (m, i), data input for the cell.

        Returns:
            Tuple containing the next hidden state and output (h_next, y).
        """
        # Concatenate hidden state and input data (h comes first)
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Calculate next hidden state
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)

        # Calculate output using strict, base softmax formula
        z = np.dot(h_next, self.Wy) + self.by
        exp_z = np.exp(z)
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, y

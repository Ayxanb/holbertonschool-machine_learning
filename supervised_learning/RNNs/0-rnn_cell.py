#!/usr/bin/env python3
"""Module providing the RNNCell class for a simple Recurrent Neural Network."""

import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""
    def __init__(self, i, h, o):
        """
        Initialize the RNN cell.

        Args:
            i (int): Dimensionality of the input data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Initialize weights with random normal distribution
        # Wh corresponds to hidden + input, so size is (h + i, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # Initialize biases as zeros
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))


def forward(self, h_prev, x_t):
    """
    Perform forward propagation for one time step.

    Args:
        h_prev (np.ndarray): Previous hidden state of shape (m, h).
        x_t (np.ndarray): Input data of shape (m, i).

    Returns:
        tuple: (h_next, y)
    """
    # Concatenate hidden state and input: (m, h + i)
    concat = np.concatenate((h_prev, x_t), axis=1)

    # Calculate next hidden state: tanh(concat @ Wh + bh)
    h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)

    # Calculate output: softmax(h_next @ Wy + by)
    z = np.dot(h_next, self.Wy) + self.by
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return h_next, y

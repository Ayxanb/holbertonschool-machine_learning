#!/usr/bin/env python3
"""Module defining the GRUCell class for a Gated Recurrent Unit."""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit."""

    def __init__(self, i, h, o):
        """
        Initializes the GRU cell.

        Args:
            i: Dimensionality of the data.
            h: Dimensionality of the hidden state.
            o: Dimensionality of the outputs.
        """
        # Initialize weights with random normal distribution in exact order
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

        # Initialize biases as zeros in exact order
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h), prev hidden state.
            x_t: numpy.ndarray of shape (m, i), data input for the cell.

        Returns:
            Tuple containing the next hidden state and output (h_next, y).
        """
        # Concatenate hidden state and input data
        concat = np.concatenate((h_prev, x_t), axis=1)

        # 1. Update gate (z)
        z_inner = np.dot(concat, self.Wz) + self.bz
        z = 1 / (1 + np.exp(-z_inner))

        # 2. Reset gate (r)
        r_inner = np.dot(concat, self.Wr) + self.br
        r = 1 / (1 + np.exp(-r_inner))

        # 3. Intermediate hidden state (h_tilde)
        # Apply reset gate to the previous hidden state
        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde_inner = np.dot(concat_r, self.Wh) + self.bh
        h_tilde = np.tanh(h_tilde_inner)

        # 4. Next hidden state
        h_next = (1 - z) * h_prev + z * h_tilde

        # 5. Output (y) using the strict base softmax formula
        z_y = np.dot(h_next, self.Wy) + self.by
        exp_y = np.exp(z_y)
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y

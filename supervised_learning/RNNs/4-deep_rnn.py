#!/usr/bin/env python3
"""Module containing the deep_rnn function."""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Args:
        rnn_cells: List of RNNCell instances of length l.
        X: numpy.ndarray of shape (t, m, i).
        h_0: numpy.ndarray of shape (l, m, h).

    Returns:
        Tuple (H, Y):
            H: numpy.ndarray of shape (t + 1, l, m, h).
            Y: numpy.ndarray of shape (t, m, o).
    """
    t, m, i = X.shape
    l, _, h = h_0.shape
    # Assuming o is inferred from the last cell's output weight shape
    o = rnn_cells[-1].by.shape[1]

    # Pre-allocate H and Y
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    # Iterate through time steps
    for step in range(t):
        # Input for the first layer is the input data
        x_t = X[step]

        # Iterate through layers
        for layer in range(l):
            # Input to the current layer is the output of the previous layer
            # (or X at the first layer)
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x_t)

            # Store the hidden state
            H[step + 1, layer] = h_next
            # The next layer's input is the hidden state of this layer
            x_t = h_next

            # If at the last layer, store the output
            if layer == l - 1:
                Y[step] = y

    return H, Y

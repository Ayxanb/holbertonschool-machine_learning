#!/usr/bin/env python3
"""Module containing the rnn function for forward propagation."""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Args:
        rnn_cell: Instance of RNNCell used for forward propagation.
        X: numpy.ndarray of shape (t, m, i) containing the data.
            t is the maximum number of time steps.
            m is the batch size.
            i is the dimensionality of the data.
        h_0: numpy.ndarray of shape (m, h), the initial hidden state.
            m is the batch size.
            h is the dimensionality of the hidden state.

    Returns:
        Tuple (H, Y):
            H: numpy.ndarray containing all of the hidden states.
            Y: numpy.ndarray containing all of the outputs.
    """
    t, m, i = X.shape
    _, h = h_0.shape

    # Pre-allocate H to store h_0 and all subsequent hidden states
    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    # List to collect outputs at each time step
    Y = []

    h_prev = h_0

    # Loop through each time step
    for step in range(t):
        # Perform forward pass for the current time step
        h_prev, y = rnn_cell.forward(h_prev, X[step])

        # Store the calculated hidden state and output
        H[step + 1] = h_prev
        Y.append(y)

    return H, np.array(Y)

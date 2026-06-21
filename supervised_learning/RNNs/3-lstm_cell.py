#!/usr/bin/env python3
"""Module defining the LSTMCell class."""

import numpy as np


class LSTMCell:
    """Represents a long short-term memory (LSTM) unit."""

    def __init__(self, i, h, o):
        """
        Initializes the LSTM cell.

        Args:
            i: Dimensionality of the data.
            h: Dimensionality of the hidden state.
            o: Dimensionality of the outputs.
        """
        # Weights initialized with random normal distribution
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

        # Biases initialized to zeros
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) prev hidden state.
            c_prev: numpy.ndarray of shape (m, h) prev cell state.
            x_t: numpy.ndarray of shape (m, i) data input.

        Returns:
            h_next, c_next, y
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f = 1 / (1 + np.exp(-(np.dot(concat, self.Wf) + self.bf)))
        # Update gate
        u = 1 / (1 + np.exp(-(np.dot(concat, self.Wu) + self.bu)))
        # Candidate cell state
        c_cand = np.tanh(np.dot(concat, self.Wc) + self.bc)
        # Output gate
        o = 1 / (1 + np.exp(-(np.dot(concat, self.Wo) + self.bo)))

        # Next cell state
        c_next = f * c_prev + u * c_cand
        # Next hidden state
        h_next = o * np.tanh(c_next)

        # Output
        z_y = np.dot(h_next, self.Wy) + self.by
        exp_y = np.exp(z_y)
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, c_next, y

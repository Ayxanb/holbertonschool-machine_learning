#!/usr/bin/env python3
"""
Module to perform forward propagation over a convolutional layer.
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a NN.

    Args:
        A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) - previous output.
        W: numpy.ndarray (kh, kw, c_prev, c_new) - kernels.
        b: numpy.ndarray (1, 1, 1, c_new) - biases.
        activation: function - activation function to be applied.
        padding: str - "same" or "valid".
        stride: tuple (sh, sw) - strides for the convolution.

    Returns:
        The output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Determine padding
    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:
        ph, pw = 0, 0

    # Apply padding
    A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                      mode='constant')

    # Calculate output dimensions
    h_out = ((h_prev + 2 * ph - kh) // sh) + 1
    w_out = ((w_prev + 2 * pw - kw) // sw) + 1

    # Initialize output volume
    Z = np.zeros((m, h_out, w_out, c_new))

    # Perform convolution
    for h in range(h_out):
        for w in range(w_out):
            # Extract the current slice (window)
            h_start, w_start = h * sh, w * sw
            h_end, w_end = h_start + kh, w_start + kw
            slice_A = A_padded[:, h_start:h_end, w_start:w_end, :]

            # Compute the convolution for all examples and kernels at once
            # slice_A: (m, kh, kw, c_prev)
            # W: (kh, kw, c_prev, c_new)
            # Result of tensordot: (m, c_new)
            Z[:, h, w, :] = np.tensordot(
                    slice_A, W, axes=([1, 2, 3], [0, 1, 2]))

    return activation(Z + b)

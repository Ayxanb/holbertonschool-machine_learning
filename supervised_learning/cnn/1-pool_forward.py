#!/usr/bin/env python3
"""
Module to perform forward propagation over a pooling layer.
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Args:
        A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) - previous output.
        kernel_shape: tuple (kh, kw) - size of the pooling kernel.
        stride: tuple (sh, sw) - strides for the pooling.
        mode: str - "max" or "avg".

    Returns:
        The output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    # Standard pooling does not typically use padding
    h_out = ((h_prev - kh) // sh) + 1
    w_out = ((w_prev - kw) // sw) + 1

    # Initialize output volume
    # Pooling keeps the number of channels the same as the input
    output = np.zeros((m, h_out, w_out, c_prev))

    # Perform pooling operation
    for h in range(h_out):
        for w in range(w_out):
            h_start, w_start = h * sh, w * sw
            h_end, w_end = h_start + kh, w_start + kw

            # Extract the window across all examples and channels
            # Shape: (m, kh, kw, c_prev)
            slice_A = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                output[:, h, w, :] = np.max(slice_A, axis=(1, 2))
            elif mode == 'avg':
                output[:, h, w, :] = np.mean(slice_A, axis=(1, 2))

    return output

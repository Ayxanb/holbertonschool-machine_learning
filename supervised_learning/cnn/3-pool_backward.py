#!/usr/bin/env python3
"""
Module to perform backward propagation over a pooling layer.
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backward propagation over a pooling layer of a neural network.

    Args:
        dA: numpy.ndarray (m, h_new, w_new, c_new) - gradient of cost w.r.t
            output of the pooling layer.
        A_prev: numpy.ndarray (m, h_prev, w_prev, c) - output of
            the previous layer.
        kernel_shape: tuple (kh, kw) - size of the pooling kernel.
        stride: tuple (sh, sw) - strides for the pooling.
        mode: str - "max" or "avg".

    Returns:
        dA_prev: partial derivatives w.r.t the previous layer.
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize the output gradient with zeros
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):  # Loop over examples
        for h in range(h_new):  # Loop over output height
            for w in range(w_new):  # Loop over output width
                for c in range(c_new):  # Loop over channels
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    if mode == 'max':
                        # Get the slice from the forward pass
                        a_prev_slice = A_prev[i, v_start:v_end,
                                              h_start:h_end, c]
                        # Create a mask where the max value was
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        # Pass the gradient only to that specific spot
                        dA_prev[i, v_start:v_end, h_start:h_end, c] += (
                            mask * dA[i, h, w, c]
                        )

                    elif mode == 'avg':
                        # Distribute the gradient equally
                        avg_gradient = dA[i, h, w, c] / (kh * kw)
                        dA_prev[i, v_start:v_end, h_start:h_end, c] += (
                            np.ones((kh, kw)) * avg_gradient
                        )

    return dA_prev

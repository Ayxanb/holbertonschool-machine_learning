#!/usr/bin/env python3
"""
Module to perform backward propagation over a convolutional layer.
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backward propagation over a convolutional layer.

    Args:
        dZ: numpy.ndarray (m, h_new, w_new, c_new) - gradient of cost w.r.t Z.
        A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) - previous output.
        W: numpy.ndarray (kh, kw, c_prev, c_new) - kernels.
        b: numpy.ndarray (1, 1, 1, c_new) - biases.
        padding: str - "same" or "valid".
        stride: tuple (sh, sw) - strides for the convolution.

    Returns:
        dA_prev, dW, db: partial derivatives w.r.t prev layer, kernels, biases.
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Determine padding
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    # Initialize gradients
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Pad inputs
    A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                      mode='constant')
    dA_padded = np.zeros_like(A_padded)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    # Define the corners of the current slice
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    # Extract the slice from the padded input
                    a_slice = A_padded[i, v_start:v_end, h_start:h_end, :]

                    # Update gradients for the window
                    dA_padded[i, v_start:v_end, h_start:h_end, :] += (
                        W[:, :, :, c] * dZ[i, h, w, c]
                    )
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

    # Strip padding from dA_prev
    if padding == "same":
        dA_prev = dA_padded[:, ph:-ph if ph != 0 else None,
                            pw:-pw if pw != 0 else None, :]
    else:
        dA_prev = dA_padded

    return dA_prev, dW, db

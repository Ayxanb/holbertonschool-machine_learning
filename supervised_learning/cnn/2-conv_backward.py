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
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + (h_prev % 2 == 0)
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + (w_prev % 2 == 0)
    else:
        ph, pw = 0, 0

    # Initialize gradients
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Pad A_prev and dA_prev
    A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                      mode='constant')
    dA_padded = np.pad(dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                       mode='constant')

    for i in range(m):  # Loop over examples
        a_prev_pad = A_padded[i]
        da_prev_pad = dA_padded[i]
        for h in range(h_new):  # Loop over vertical output
            for w in range(w_new):  # Loop over horizontal output
                for c in range(c_new):  # Loop over channels
                    h_start, w_start = h * sh, w * sw
                    h_end, w_end = h_start + kh, w_start + kw

                    # Extract current slice
                    a_slice = a_prev_pad[h_start:h_end, w_start:w_end, :]

                    # Update gradients
                    da_prev_pad[h_start:h_end, w_start:w_end, :] += (
                        W[:, :, :, c] * dZ[i, h, w, c]
                    )
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        # Remove padding from dA_prev
        if padding == "same":
            dA_prev[i, :, :, :] = da_prev_pad[ph:-ph, pw:-pw, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad

    return dA_prev, dW, db

#!/usr/bin/env python3
import numpy as np
"""
This module contains `convolve_grayscale`  a simple convolution
implementation for grayscale images with padding & stride.
"""


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs convolution on a batch of grayscale images.

    Parameters
    ----------
    images : numpy.ndarray, shape (m, h, w)
        Batch of m grayscale images (height h, width w).
    kernel : numpy.ndarray, shape (kh, kw)
        Convolution kernel (filter).
    padding : 'valid', 'same', or tuple (ph, pw), default 'same'
        - 'valid': no padding
        - 'same': padding to keep output size close to input (when stride=1)
        - tuple: custom padding per side
    stride : tuple (sh, sw), default (1, 1)
        Stride along height and width.

    Returns
    -------
    numpy.ndarray, shape (m, out_h, out_w)
        Convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding amounts (top/bottom, left/right)
    if padding == 'valid':
        pad_top = pad_bottom = pad_left = pad_right = 0
    elif padding == 'same':
        pad_top = (kh - 1) // 2
        pad_bottom = kh - 1 - pad_top
        pad_left = (kw - 1) // 2
        pad_right = kw - 1 - pad_left
    else:  # tuple (ph, pw)  padding per sid
        ph, pw = padding
        pad_top = pad_bottom = ph
        pad_left = pad_right = pw

    # Zero-pad the images (symmetric or asymmetric as needed)
    padded = np.pad(
        images,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )

    # Calculate output size
    padded_h = padded.shape[1]
    padded_w = padded.shape[2]
    out_h = (padded_h - kh) // sh + 1
    out_w = (padded_w - kw) // sw + 1

    output = np.zeros((m, out_h, out_w))

    # ONLY TWO for loops  over output positions i and j (as hinted)
    for i in range(out_h):
        for j in range(out_w):
            # Extract the kernel-sized window from padded images
            row_start = i * sh
            col_start = j * sw
            window = padded[:, row_start:row_start+kh, col_start:col_start+kw]
            # Convolve: sum over kernel dimensions for every image at once
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output

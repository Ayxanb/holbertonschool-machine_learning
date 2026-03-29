#!/usr/bin/env python3
import numpy as np
"""
This module contains `convolve_grayscale_padding`  a simple convolution
implementation for grayscale images with custom padding.
"""


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs convolution on a batch of grayscale images with custom padding.

    Parameters
    ----------
    images : numpy.ndarray, shape (m, h, w)
        Batch of m grayscale images (height h, width w).
    kernel : numpy.ndarray, shape (kh, kw)
        Convolution kernel (filter).
    padding : tuple (ph, pw)
        Padding amount: ph rows on top/bottom, pw columns on left/right.

    Returns
    -------
    numpy.ndarray, shape (m, out_h, out_w)
        Convolved images (output size depends on padding).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Zero-pad the images (symmetric padding, no extra loops)
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Output size for the given padding
    out_h = h + 2 * ph - kh + 1
    out_w = w + 2 * pw - kw + 1
    output = np.zeros((m, out_h, out_w))

    # Only TWO for loops (exactly like valid/same versions)
    for ky in range(kh):
        for kx in range(kw):
            output += padded[:, ky:ky + out_h, kx:kx + out_w] * kernel[ky, kx]

    return output

#!/usr/bin/env python3
import numpy as np
"""
This module contains `convolve_grayscale_same` a simple "same" convolution
implementation for grayscale images (used for learning CNN basics).
"""

def convolve_grayscale_same(images, kernel):
    """
    Performs same convolution on a batch of grayscale images.

    Parameters
    ----------
    images : numpy.ndarray, shape (m, h, w)
        Batch of m grayscale images (height h, width w).
    kernel : numpy.ndarray, shape (kh, kw)
        Convolution kernel (filter).

    Returns
    -------
    numpy.ndarray, shape (m, h, w)
        Convolved images using same mode (output size = input size,
        with zero-padding where necessary).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Padding needed for "same" mode (standard CNN style)
    # pad_top/left gets the floor half, pad_bottom/right gets the rest
    pad_top = (kh - 1) // 2
    pad_bottom = kh - 1 - pad_top
    pad_left = (kw - 1) // 2
    pad_right = kw - 1 - pad_left

    # Zero-pad the images (no extra loops!)
    padded = np.pad(
        images,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )

    # Output size is exactly the same as input
    out_h = h
    out_w = w
    output = np.zeros((m, out_h, out_w))

    # Only TWO for loops (exactly like the valid version)
    for ky in range(kh):
        for kx in range(kw):
            output += padded[:, ky:ky + out_h, kx:kx + out_w] * kernel[ky, kx]

    return output

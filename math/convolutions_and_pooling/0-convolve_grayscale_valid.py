#!/usr/bin/env python3
import numpy as np
"""
This module contains `convolve_grayscale_valid` — a simple valid convolution
implementation for grayscale images (used for learning CNN basics).
"""


def convolve_grayscale_valid(images, kernel):
    """
    Performs valid convolution on a batch of grayscale images.

    Parameters
    ----------
    images : numpy.ndarray, shape (m, h, w)
        Batch of m grayscale images (height h, width w).
    kernel : numpy.ndarray, shape (kh, kw)
        Convolution kernel (filter).

    Returns
    -------
    numpy.ndarray, shape (m, out_h, out_w)
        Convolved images using valid mode (no padding).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    output = np.zeros((m, out_h, out_w))
    for ky in range(kh):
        for kx in range(kw):
            output += images[:, ky:ky + out_h, kx:kx + out_w] * kernel[ky, kx]
    return output

#!/usr/bin/env python3
import numpy as np

"""
This module contains `convolve_grayscale_same` function.
"""

def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Standard 'same' padding formula: p = (k - 1) / 2
    # We use // 2 to handle the integer division
    ph = kh // 2
    pw = kw // 2

    # Apply zero padding to the height and width dimensions
    # Shape: (m, h + 2*ph, w + 2*pw)
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 
                    mode='constant', constant_values=0)

    # Initialize the output array with the same shape as input
    convolved = np.zeros((m, h, w))

    # Perform convolution using exactly two loops (over the kernel dimensions)
    # This vectorizes the operation across all images (m) and all pixels (h, w)
    for i in range(kh):
        for j in range(kw):
            # Slice the padded image and multiply by the corresponding kernel weight
            # The slice starts at the current kernel index and takes h/w pixels
            convolved += padded[:, i:i + h, j:j + w] * kernel[i, j]

    return convolved

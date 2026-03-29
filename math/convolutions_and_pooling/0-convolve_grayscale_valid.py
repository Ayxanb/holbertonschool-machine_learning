#!/usr/bin/env python3
import numpy as np
"""
This module contains `convolve_grayscale_valid` a simple "valid" convolution
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
        Convolution kernel (filter) to apply.

    Returns
    -------
    numpy.ndarray, shape (m, out_h, out_w)
        Convolved images where out_h = h - kh + 1 and out_w = w - kw + 1.
    """

    # Extract dimensions: number of images, height, and width
    m, h, w = images.shape
    
    # Extract kernel dimensions: height and width
    kh, kw = kernel.shape
    
    # Calculate output dimensions for 'valid' convolution (no padding)
    # The kernel must fit entirely within the image boundaries
    out_h = h - kh + 1
    out_w = w - kw + 1
    
    # Initialize the output tensor with zeros
    output = np.zeros((m, out_h, out_w))

    # Optimization: Instead of looping over every pixel in the output,
    # we loop over every 'weight' in the kernel.
    for ky in range(kh):
        for kx in range(kw):
            # images[:, ky:ky + out_h, kx:kx + out_w] extracts a 'slice' of 
            # the entire batch of images that corresponds to the current 
            # kernel element (ky, kx).
            
            # We multiply this entire 3D slice by a single scalar (the kernel weight)
            # and add it to the running total in the output array.
            output += images[:, ky:ky + out_h, kx:kx + out_w] * kernel[ky, kx]
            
    return output

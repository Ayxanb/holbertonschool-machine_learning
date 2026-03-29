#!/usr/bin/env python3
"""
This module contains the function `pool` which performs
pooling on a batch of multi-channel images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on a batch of multi-channel images.

    Parameters
    ----------
    images : numpy.ndarray, shape (m, h, w, c)
        m : number of images in the batch
        h : height of the images in pixels
        w : width of the images in pixels
        c : number of channels
    kernel_shape : tuple of (kh, kw)
        kh : height of the pooling kernel
        kw : width of the pooling kernel
    stride : tuple of (sh, sw)
        sh : stride for the height
        sw : stride for the width
    mode : str, 'max' or 'avg'
        'max' : max pooling
        'avg' : average pooling

    Returns
    -------
    numpy.ndarray, shape (m, out_h, out_w, c)
        The pooled images.
    """
    # Extract dimensions
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # 1. Calculate Output Spatial Dimensions
    # Pooling is naturally a 'valid' operation (no padding by default)
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    # Initialize output: (m images, out_height, out_width, c channels)
    output = np.zeros((m, out_h, out_w, c))

    # 2. Perform Pooling with Two Loops
    # We loop over output height and width.
    # Since pooling is independent per channel and image,
    # we use NumPy's vectorization to process (m, c) simultaneously.
    for i in range(out_h):
        for j in range(out_w):
            # Define the current pooling window
            h_start, w_start = i * sh, j * sw
            h_end, w_end = h_start + kh, w_start + kw

            # Extract window: (m, kh, kw, c)
            window = images[:, h_start:h_end, w_start:w_end, :]

            # 3. Apply the pooling operation based on mode
            if mode == 'max':
                # Maximize across the spatial kernel axes (1 and 2)
                output[:, i, j, :] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                # Average across the spatial kernel axes (1 and 2)
                output[:, i, j, :] = np.mean(window, axis=(1, 2))

    return output

#!/usr/bin/env python3
"""
This module contains the function `convolve_grayscale_valid`.
It implements a "valid" convolution for grayscale image batches
using an optimized kernel-looping approach.
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on a batch of grayscale images.

    This implementation uses a vectorized approach by looping over the
    dimensions of the kernel rather than the pixels of the output image.
    By slicing the input images and multiplying by individual kernel weights,
    it leverages NumPy's broadcasting to process the entire batch and all
    spatial locations simultaneously.

    Parameters
    ----------
    images : numpy.ndarray
        A 3D array of shape (m, h, w) representing a batch of images.
        m : The number of images in the batch.
        h : The height of each image in pixels.
        w : The width of each image in pixels.
    kernel : numpy.ndarray
        A 2D array of shape (kh, kw) representing the convolution filter.
        kh : The height of the kernel.
        kw : The width of the kernel.

    Returns
    -------
    numpy.ndarray
        A 3D array of shape (m, out_h, out_w) containing the convolved
        images. The output dimensions are calculated as:
        out_h = h - kh + 1
        out_w = w - kw + 1
    """
    # m: batch size, h: image height, w: image width
    m, h, w = images.shape

    # kh: kernel height, kw: kernel width
    kh, kw = kernel.shape

    # Valid convolution: output size is reduced because no padding is used.
    # The kernel must stay fully within the original image boundaries.
    out_h = h - kh + 1
    out_w = w - kw + 1

    # Initialize the output tensor. All images in the batch are processed.
    output = np.zeros((m, out_h, out_w))

    # Perform the convolution by iterating over the kernel's spatial weights.
    # For every weight in the (kh, kw) kernel, we take a corresponding
    # (out_h, out_w) slice of the image batch and accumulate the product.
    for ky in range(kh):
        for kx in range(kw):
            # images[:, ky:ky + out_h, kx:kx + out_w] extracts a 'slice' of
            # the entire batch that is offset by the current kernel index.
            # This 'slice' has the same shape as the 'output' array.
            
            # Multiply the batch slice by the specific scalar weight at
            # kernel[ky, kx] and add the result to the output accumulator.
            output += images[:, ky:ky + out_h, kx:kx + out_w] * kernel[ky, kx]

    return output

#!/usr/bin/env python3
"""
This module contains the function `convolve_grayscale_padding`.
It implements a convolution for grayscale images using custom
zero-padding provided by the user.
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on a batch of grayscale images with custom padding.

    This function applies a convolution kernel to a batch of images after
    applying zero-padding to the spatial dimensions. The operation is
    optimized using NumPy broadcasting by iterating over the kernel's
    spatial dimensions rather than the output's pixels.

    Parameters
    ----------
    images : numpy.ndarray
        A 3D array of shape (m, h, w) containing the image batch.
        m : The number of images.
        h : The height of each image in pixels.
        w : The width of each image in pixels.
    kernel : numpy.ndarray
        A 2D array of shape (kh, kw) containing the convolution filter.
        kh : The height of the kernel.
        kw : The width of the kernel.
    padding : tuple
        A tuple of (ph, pw) representing the padding for the height
        and width, respectively. ph rows are added to the top and bottom,
        and pw columns are added to the left and right.

    Returns
    -------
    numpy.ndarray
        A 3D array of shape (m, out_h, out_w) containing the convolved
        images. The output dimensions are:
        out_h = h + (2 * ph) - kh + 1
        out_w = w + (2 * pw) - kw + 1
    """
    # m: batch size, h: image height, w: image width
    m, h, w = images.shape
    # kh: kernel height, kw: kernel width
    kh, kw = kernel.shape
    # ph: height padding, pw: width padding
    ph, pw = padding

    # Apply symmetric zero-padding to the height and width dimensions.
    # The batch dimension (axis 0) remains unpadded.
    # Resulting shape: (m, h + 2*ph, w + 2*pw)
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Calculate the resulting output dimensions.
    # These represent how many times the kernel can fit spatially.
    out_h = h + 2 * ph - kh + 1
    out_w = w + 2 * pw - kw + 1

    # Initialize the output tensor.
    output = np.zeros((m, out_h, out_w))

    # Perform convolution by iterating over the kernel's spatial dimensions.
    # This method treats the convolution as a weighted sum of shifted images.
    for ky in range(kh):
        for kx in range(kw):
            # Extract a slice of the padded batch that is offset by the
            # current kernel coordinates (ky, kx).
            # Each slice has the spatial shape (out_h, out_w).

            # The slice is multiplied by the specific scalar weight at
            # kernel[ky, kx] and accumulated into the output.
            output += padded[:, ky:ky + out_h, kx:kx + out_w] * kernel[ky, kx]

    return output

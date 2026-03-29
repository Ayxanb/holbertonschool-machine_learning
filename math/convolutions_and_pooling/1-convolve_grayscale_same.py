#!/usr/bin/env python3
"""
This module contains the function `convolve_grayscale_same`.
It performs a "same" convolution on a batch of grayscale images.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a 'same' convolution on a batch of grayscale images.

    A 'same' convolution ensures that the output spatial dimensions
    (height and width) are identical to the input spatial dimensions.
    This is achieved by padding the input images with zeros.

    This function utilizes a highly optimized vectorized approach.
    Instead of iterating over every pixel in every image, it iterates
    over the dimensions of the kernel. For each kernel weight, it
    multiplies a shifted slice of the entire image batch, accumulating
    the results into the output array.

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

    Returns
    -------
    numpy.ndarray
        A 3D array of shape (m, h, w) containing the convolved images.
        The output size matches the input size regardless of kernel size.
    """
    # m: batch size, h: original height, w: original width
    m, h, w = images.shape
    # kh: kernel height, kw: kernel width
    kh, kw = kernel.shape

    # Calculate symmetric padding for 'same' convolution.
    # ph/pw represent the number of pixels added to each side.
    # Example: A 3x3 kernel (kh=3) results in ph = 3 // 2 = 1 pixel
    # of padding on the top and 1 on the bottom.
    ph = kh // 2
    pw = kw // 2

    # Apply zero padding to the height (axis 1) and width (axis 2).
    # The batch (axis 0) is left unpadded.
    # padded shape: (m, h + 2*ph, w + 2*pw)
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant', constant_values=0)

    # Initialize the output tensor with the same spatial shape as the input.
    convolved = np.zeros((m, h, w))

    # Loop through the kernel height and width.
    # We use exactly two loops as required by the constraint.
    for i in range(kh):
        for j in range(kw):
            # Extract a slice of the padded images starting at (i, j).
            # Each slice is exactly the size of the original image (h, w).
            # By iterating i and j, we effectively 'slide' the image
            # under the kernel.

            # This line performs the multiplication for all 'm' images
            # and all pixels simultaneously using NumPy broadcasting.
            convolved += padded[:, i:i + h, j:j + w] * kernel[i, j]

    return convolved

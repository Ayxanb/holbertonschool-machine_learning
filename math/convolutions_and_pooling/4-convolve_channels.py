#!/usr/bin/env python3
"""
This module contains the function `convolve_channels`.
It performs a multi-channel convolution on a batch of images.
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on a batch of images with multiple channels.

    This function processes a batch of images (m, h, w, c) using a 3D
    kernel (kh, kw, c). It uses a window-based approach to handle
    strides and utilizes NumPy's broadcasting to perform the element-wise
    multiplication and summation across all images and channels
    simultaneously.

    Parameters
    ----------
    images : numpy.ndarray
        A 4D array of shape (m, h, w, c) containing the image batch.
        m : The number of images.
        h : The height of each image in pixels.
        w : The width of each image in pixels.
        c : The number of channels in the image.
    kernel : numpy.ndarray
        A 3D array of shape (kh, kw, c) containing the filter.
        kh : The height of the kernel.
        kw : The width of the kernel.
        c : The number of channels (must match the input images).
    padding : str or tuple
        If 'same', padding is calculated so the output has dimensions
        ceil(input / stride). If 'valid', no padding is used.
        If a tuple (ph, pw), ph/pw are applied to both sides of the H/W axes.
    stride : tuple
        A tuple of (sh, sw) containing the stride for the height and width.

    Returns
    -------
    numpy.ndarray
        A 3D array of shape (m, out_h, out_w) containing the convolved images.
    """
    # Extract image and kernel dimensions
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    # --- 1. Determine Padding Amounts ---
    if padding == 'same':
        # Calculate padding to maintain 'same' output size relative to stride
        ph = int(np.ceil(((sh * (h - 1)) - h + kh) / 2))
        pw = int(np.ceil(((sw * (w - 1)) - w + kw) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        # Custom padding from tuple (ph, pw)
        ph, pw = padding

    # --- 2. Apply Zero Padding ---
    # We pad only the height (axis 1) and width (axis 2) dimensions.
    # The batch (axis 0) and channel (axis 3) dimensions are not padded.
    images_padded = np.pad(images,
                           ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    # --- 3. Calculate Output Spatial Dimensions ---
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    # Initialize the output tensor.
    # Note: One kernel applied to multiple channels
    # results in one output depth.
    output = np.zeros((m, out_h, out_w))

    # --- 4. Perform Convolution ---
    # We iterate over the output spatial dimensions (i, j).
    for i in range(out_h):
        for j in range(out_w):
            # Determine the window boundaries in the padded images
            h_start = i * sh
            w_start = j * sw

            # Extract the 4D window across all images and all channels.
            # window shape: (m, kh, kw, c)
            window = images_padded[:, h_start:h_start + kh,
                                   w_start:w_start + kw, :]

            # Multiply the window
            # by the kernel (broadcasting across batch 'm').
            # Sum across axis 1 (kh), 2 (kw), and 3 (c).
            # This collapses the local neighborhood
            # into a single scalar per image.
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2, 3))

    return output

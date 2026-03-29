#!/usr/bin/env python3
"""
This module contains the function `convolve_grayscale`.
It performs a convolution on a batch of grayscale images, supporting
'same' padding, 'valid' padding, custom padding, and custom strides.
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on a batch of grayscale images.

    This function applies a convolution kernel to a batch of images. It
    calculates padding based on the desired mode and slides the kernel
    across the images based on the specified stride. The operation is
    vectorized across the batch of images.

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
    padding : str or tuple
        If 'same', padding is calculated so the output has dimensions
        ceil(input / stride). If 'valid', no padding is used. If a
        tuple (ph, pw), ph is the height padding and pw is the width
        padding applied to each side.
    stride : tuple
        A tuple of (sh, sw) containing the stride for the height and
        width, respectively.

    Returns
    -------
    numpy.ndarray
        A 3D array of shape (m, out_h, out_w) containing the convolved
        images.
    """
    # m: batch size, h: input height, w: input width
    m, h, w = images.shape
    # kh: kernel height, kw: kernel width
    kh, kw = kernel.shape
    # sh: stride height, sw: stride width
    sh, sw = stride

    # --- 1. Determine Padding Amounts ---
    if padding == 'same':
        # 'Same' padding ensures the filter covers the entire input.
        # The output size will be ceil(h / sh) and ceil(w / sw).
        # We use the standard formula to find the required zero-padding.
        pad_h = int(np.ceil(((sh * (h - 1)) - h + kh) / 2))
        pad_w = int(np.ceil(((sw * (w - 1)) - w + kw) / 2))
    elif padding == 'valid':
        pad_h, pad_w = 0, 0
    else:
        # User provides custom padding as (ph, pw)
        pad_h, pad_w = padding

    # --- 2. Apply Zero Padding ---
    # np.pad adds zeros to the top/bottom (axis 1) and left/right (axis 2).
    # images_padded shape: (m, h + 2*pad_h, w + 2*pad_w)
    images_padded = np.pad(images,
                           ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant', constant_values=0)

    # --- 3. Calculate Output Dimensions ---
    # The output size is the number of times the kernel 'fits' into the
    # padded image while jumping by the stride.
    out_h = (h + 2 * pad_h - kh) // sh + 1
    out_w = (w + 2 * pad_w - kw) // sw + 1

    # Initialize the output tensor.
    output = np.zeros((m, out_h, out_w))

    # --- 4. Perform Convolution ---
    # We iterate over the output dimensions (i, j).
    # This loop is necessary to handle strides effectively.
    for i in range(out_h):
        for j in range(out_w):
            # Calculate the top-left corner of the current window.
            h_start = i * sh
            w_start = j * sw

            # Extract the window for all m images simultaneously.
            # Window shape: (m, kh, kw)
            window = images_padded[:, h_start:h_start + kh,
                                   w_start:w_start + kw]

            # Element-wise multiply the kernel against all windows in
            # the batch. Sum the results across the spatial axes
            # (1 and 2) to get a single value per image.
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output

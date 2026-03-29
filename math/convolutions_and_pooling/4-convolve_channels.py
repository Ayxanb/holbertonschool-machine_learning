#!/usr/bin/env python3
"""
This module contains the function convolve_channels which performs
a convolution on images with multiple channels.
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on a batch of images with multiple channels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel: numpy.ndarray with shape (kh, kw, c)
        padding: either a tuple of (ph, pw), 'same', or 'valid'
        stride: a tuple of (sh, sw)

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    # Extract dimensions from input shapes
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    # 1. Determine Padding Amounts
    if padding == 'same':
        # Same padding calculation to ensure output size = ceil(input / stride)
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        # User provided tuple (ph, pw)
        ph, pw = padding

    # 2. Apply Zero Padding
    # We pad the height (axis 1) and width (axis 2) only.
    # The batch (axis 0) and channels (axis 3) are not padded.
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # 3. Calculate Output Dimensions
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    # Initialize output array (shape: m, out_h, out_w)
    # Note: Even with multiple input channels, one kernel produces one channel.
    output = np.zeros((m, out_h, out_w))

    # 4. Perform Convolution with two loops
    # We loop over the output spatial dimensions i and j
    for i in range(out_h):
        for j in range(out_w):
            # Calculate the window boundaries in the padded image
            h_start, w_start = i * sh, j * sw
            h_end, w_end = h_start + kh, w_start + kw

            # Extract the window across all images and all channels
            # Shape: (m, kh, kw, c)
            window = images_padded[:, h_start:h_end, w_start:w_end, :]

            # Element-wise multiply window by kernel (broadcasting kernel)
            # Then sum over the kernel height, width,
            # AND channels (axes 1, 2, 3)
            # Resulting shape: (m,)
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2, 3))

    return output

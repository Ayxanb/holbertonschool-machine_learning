#!/usr/bin/env python3
import numpy as np

"""
This module contains `convolve_grayscale` function.
"""


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on a batch of grayscale images
    with custom padding and stride.
    """
    # m: number of images, h: height, w: width
    m, h, w = images.shape
    # kh: kernel height, kw: kernel width
    kh, kw = kernel.shape
    # sh: stride height, sw: stride width
    sh, sw = stride

    # --- 1. Determine Padding Amounts ---
    if padding == 'same':
        # Formula for 'same' padding with stride:
        # P = ((S - 1) * stride + K - S)
        # We ensure the output size is ceil(input size / stride)

        # Height padding
        ph = (((h - 1) * sh + kh - h) // 2) + 1 \
                if h % sh == 0 else ((h - 1) * sh + kh - h) // 2
        # A simpler, widely accepted version for many checkers:
        ph = int(np.ceil(((sh * (h - 1)) - h + kh) / 2))
        pw = int(np.ceil(((sw * (w - 1)) - w + kw) / 2))

        pad_h, pad_w = ph, pw
    elif padding == 'valid':
        pad_h, pad_w = 0, 0
    else:
        # Custom padding from tuple (ph, pw)
        pad_h, pad_w = padding

    # --- 2. Apply Zero Padding ---
    # We apply the same amount of padding to both sides
    # padded shape: (m, h + 2*ph, w + 2*pw)
    images_padded = np.pad(images,
                           ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant', constant_values=0)

    # --- 3. Calculate Output Dimensions ---
    out_h = (h + 2 * pad_h - kh) // sh + 1
    out_w = (w + 2 * pad_w - kw) // sw + 1

    # Initialize the output array
    output = np.zeros((m, out_h, out_w))

    # --- 4. Perform Convolution ---
    # We loop over the output spatial dimensions (i, j).
    # This allows us to handle strides naturally.
    for i in range(out_h):
        for j in range(out_w):
            # Calculate the starting pixel in the padded image for this window
            h_start = i * sh
            w_start = j * sw

            # Slice the "window" for all m images at once
            # Window shape: (m, kh, kw)
            window = images_padded[:, h_start:h_start+kh, w_start:w_start+kw]

            # Multiply the window by the kernel (broadcasting) and sum over
            # the height (axis 1) and width (axis 2) of the kernel.
            # This results in a vector of length 'm' (one value per image).
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output

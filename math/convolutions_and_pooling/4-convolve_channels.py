#!/usr/bin/env python3
"""
This module contains the function `convolve_grayscale`.
It provides a flexible convolution implementation capable of handling
various padding schemes and strides for grayscale image batches.
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on a batch of grayscale images.

    This function utilizes a window-based approach to accommodate strides
    greater than 1. While looping over the output spatial dimensions (i, j),
    it leverages NumPy's broadcasting to process the entire batch of 'm'
    images in a single operation per window.

    Mathematical Logic:
    - Padding: Zeroes are added to ensure the kernel can traverse the image.
    - Stride: Controls the step size of the kernel. A stride of 2 halves
      the output resolution.
    - Vectorization: The multiplication and summation occur across the
      entire batch (axis 0) simultaneously.

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
        If 'same', padding is calculated to maintain input spatial dimensions
        (output = input / stride). If 'valid', no padding is applied.
        If a tuple (ph, pw), ph/pw are applied to both sides of the H/W axes.
    stride : tuple
        A tuple of (sh, sw) containing the stride for the height and width.

    Returns
    -------
    numpy.ndarray
        A 3D array of shape (m, out_h, out_w) containing the convolved images.
    """
    # m: batch size, h: input height, w: input width
    m, h, w = images.shape
    # kh: kernel height, kw: kernel width
    kh, kw = kernel.shape
    # sh: stride height, sw: stride width
    sh, sw = stride

    # --- 1. Determine Padding Amounts ---
    if padding == 'same':
        # Standard formula for 'same' padding to ensure the output
        # spatial size is ceil(input_size / stride).
        pad_h = int(np.ceil(((sh * (h - 1)) - h + kh) / 2))
        pad_w = int(np.ceil(((sw * (w - 1)) - w + kw) / 2))
    elif padding == 'valid':
        pad_h, pad_w = 0, 0
    else:
        # Custom padding provided as a tuple (ph, pw)
        pad_h, pad_w = padding

    # --- 2. Apply Zero Padding ---
    # We apply the calculated padding symmetrically to the H and W axes.
    # padded shape: (m, h + 2*pad_h, w + 2*pad_w)
    images_padded = np.pad(images,
                           ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant', constant_values=0)

    # --- 3. Calculate Output Dimensions ---
    # The output height and width are determined by how many windows
    # fit into the padded image given the step size (stride).
    out_h = (h + 2 * pad_h - kh) // sh + 1
    out_w = (w + 2 * pad_w - kw) // sw + 1

    # Initialize the output tensor.
    output = np.zeros((m, out_h, out_w))

    # --- 4. Perform Convolution ---
    # We iterate over the output dimensions (i, j).
    # This loop allows us to jump across the image according to the stride.
    for i in range(out_h):
        for j in range(out_w):
            # Calculate the top-left starting corner for the current stride
            h_start = i * sh
            w_start = j * sw

            # Extract the 'window' across the entire batch (m images).
            # window shape: (m, kh, kw)
            window = images_padded[:, h_start:h_start + kh,
                                   w_start:w_start + kw]

            # Multiply the window by the kernel (broadcasting kernel weights
            # across the batch) and sum the height/width axes (1 and 2).
            # This yields a result vector of length 'm'.
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output

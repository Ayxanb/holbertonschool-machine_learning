#!/usr/bin/env python3
"""
This module provides utility functions for image preprocessing.

It includes operations for color space and intensity transformations,
specifically focusing on random contrast adjustments for data augmentation.
"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of a 3D image tensor.

    Args:
        image (tf.Tensor): A 3D tensor of shape [height, width, channels]
            representing the image.
        lower (float): The lower bound of the random contrast factor range.
        upper (float): The upper bound of the random contrast factor range.

    Returns:
        tf.Tensor: The contrast-adjusted image tensor.
    """
    return tf.image.random_contrast(
            image,
            lower=lower,
            upper=upper
    )

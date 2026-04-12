#!/usr/bin/env python3
"""
This module provides utility functions for image preprocessing.

It includes operations for pixel-level intensity transformations, specifically
focusing on random brightness adjustments to enhance model robustness.
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of a 3D image tensor.

    Args:
        image (tf.Tensor): A 3D tensor of shape [height, width, channels]
            representing the image.
        max_delta (float): The maximum amount the image should be
            brightened or darkened. Must be non-negative.

    Returns:
        tf.Tensor: The brightness-adjusted image tensor.
    """
    return tf.image.random_brightness(image, max_delta=max_delta)

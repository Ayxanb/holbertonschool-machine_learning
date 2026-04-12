#!/usr/bin/env python3
"""
This module provides utility functions for image preprocessing.

It includes operations for color space transformations, specifically focusing
on hue adjustments to simulate different lighting or color variations.
"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of a 3D image tensor.

    Args:
        image (tf.Tensor): A 3D tensor of shape [height, width, channels]
            representing the image.
        delta (float): The amount by which to adjust the hue. This value
            is added to the hue channel in the range [-1, 1].

    Returns:
        tf.Tensor: The hue-adjusted image tensor.
    """
    return tf.image.adjust_hue(image, delta)

#!/usr/bin/env python3
"""
This module provides utility functions for image preprocessing.
It includes operations for geometric transformations, specifically focusing
on horizontal flipping of image tensors to aid in data augmentation or
image manipulation tasks.
"""

import tensorflow as tf


def flip_image(image):
    """
    Flips a 3D image tensor horizontally (left to right).

    Args:
        image (tf.Tensor): A 3D tensor of shape [height, width, channels]
            representing the image.

    Returns:
        tf.Tensor: The horizontally flipped image tensor.
    """
    return tf.image.flip_left_right(image)

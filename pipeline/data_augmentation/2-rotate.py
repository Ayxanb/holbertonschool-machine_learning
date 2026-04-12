#!/usr/bin/env python3
"""
This module provides utility functions for image preprocessing.

It includes operations for geometric transformations, specifically focusing
on rotating image tensors by 90-degree increments for data augmentation.
"""

import tensorflow as tf


def rotate_image(image):
    """
    Rotates a 3D image tensor by 90 degrees counter-clockwise.

    Args:
        image (tf.Tensor): A 3D tensor of shape [height, width, channels]
            representing the image to rotate.

    Returns:
        tf.Tensor: The rotated image tensor.
    """
    return tf.image.rot90(image, k=1)

#!/bin/usr/env python3
"""
This module provides utility functions for image preprocessing.

It includes operations for geometric transformations, specifically focusing
on random cropping of image tensors to aid in data augmentation.
"""

import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of a 3D image tensor.

    Args:
        image (tf.Tensor): A 3D tensor of shape [height, width, channels]
            representing the image.
        size (tuple): A tuple (target_height, target_width, channels)
            specifying the dimensions of the crop.

    Returns:
        tf.Tensor: The randomly cropped image tensor.
    """
    return tf.image.random_crop(image, size=size)

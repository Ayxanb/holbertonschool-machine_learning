#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

"""
This module contains a class NST that performs tasks for neural style transfer
"""


class NST:
    """
    A class to perform tasks for Neural Style Transfer.
    """

    # Public class attributes
    style_layers = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1'
        ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for NST.
        """

        # Validate style_image
        if not isinstance(style_image, np.ndarray) or \
                len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        # Validate content_image
        if not isinstance(content_image, np.ndarray) or \
                len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                 "content_image must be a numpy.ndarray with shape (h, w, 3)"
                )

        # Validate alpha
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        # Validate beta
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Set instance attributes
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels.
        """

        # Validate image
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                    "image must be a numpy.ndarray with shape (h, w, 3)")

        # Calculate new dimensions
        h, w, _ = image.shape
        scale_factor = 512.0 / max(h, w)
        h_new = int(h * scale_factor)
        w_new = int(w * scale_factor)

        # Convert to tensor and expand dimensions to (1, h_new, w_new, 3)
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, axis=0)

        # Resize the image using bicubic interpolation
        resized_image = tf.image.resize(
            image_tensor,
            size=[h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )

        # Rescale pixel values from [0, 255] to [0, 1]
        scaled_image = resized_image / 255.0

        # Clip values to ensure they stay exactly within [0, 1]
        scaled_image = tf.clip_by_value(scaled_image, 0.0, 1.0)

        return scaled_image

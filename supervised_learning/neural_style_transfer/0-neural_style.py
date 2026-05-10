#!/usr/bin/env python3
"""
Neural Style Transfer Module

This module contains the `NST` class, which provides the foundational setup
and preprocessing tasks required for performing neural style transfer (NST).
It utilizes TensorFlow and NumPy to manipulate and scale images appropriately
for input into a deep neural network (such as VGG19).
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    A class to perform tasks for Neural Style Transfer.

    This class sets up the necessary style and content layers to be extracted
    from a pre-trained model and handles the initialization, validation, and
    preprocessing of the style and content images.

    Attributes:
        style_layers (list): A list of strings identifying the network layers
            used to extract the style features. Default is ['block1_conv1',
            'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'].
        content_layer (str): A string identifying the network layer used to
            extract the content features. Default is 'block5_conv2'.
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
        Class constructor for NST. Initializes and preprocesses the style
        and content images, and sets the weighting factors for the costs.

        Args:
            style_image (numpy.ndarray): The image used as a style reference.
                Must have the shape (h, w, 3).
            content_image (numpy.ndarray):
                The image used as a content reference.
                Must have the shape (h, w, 3).
            alpha (float or int, optional): The weight applied to the content
                cost during optimization. Defaults to 1e4.
            beta (float or int, optional): The weight applied to the style
                cost during optimization. Defaults to 1.

        Raises:
            TypeError:
                If `style_image` is not a numpy.ndarray of shape (h, w, 3).
            TypeError:
                If `content_image` is not a numpy.ndarray of shape (h, w, 3).
            TypeError:
                If `alpha` is not a non-negative number.
            TypeError:
                If `beta` is not a non-negative number.
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
        Rescales an image such that its pixel values are between 0 and 1,
        and its largest side is exactly 512 pixels.
        The aspect ratio is maintained.

        The image is converted to a TensorFlow tensor, expanded to include a
        batch dimension, resized using bicubic interpolation, and its values
        are clipped to ensure they strictly fall within the [0.0, 1.0] range.

        Args:
            image
                (numpy.ndarray): The image to be scaled. Must have the shape
                (h, w, 3) and original pixel values typically in the range
                [0, 255]

        Returns:
            tf.Tensor: The scaled image tensor with shape (1, h_new, w_new, 3)
                where max(h_new, w_new) == 512.

        Raises:
            TypeError: If `image` is not a numpy.ndarray of shape (h, w, 3).
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

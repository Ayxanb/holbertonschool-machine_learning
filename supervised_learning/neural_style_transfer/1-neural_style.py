#!/usr/bin/env python3
"""
Neural Style Transfer Module

This module contains the `NST` class, which provides the foundational setup
for neural style transfer, including image preprocessing and model loading.
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    A class to perform tasks for Neural Style Transfer.

    Attributes:
        style_layers (list): VGG19 layers used to extract style features.
        content_layer (str): VGG19 layer used to extract content features.
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

        Args:
            style_image (np.ndarray): Image used as a style reference.
            content_image (np.ndarray): Image used as a content reference.
            alpha (float): Weight for content cost.
            beta (float): Weight for style cost.

        Raises:
            TypeError: If images are not np.ndarray of shape (h, w, 3).
            TypeError: If alpha or beta are not non-negative numbers.
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        # Initialize the model using the load_model method
        self.model = self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image to 512px max side and normalizes pixels to [0, 1].
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        scale = 512.0 / max(h, w)
        h_new, w_new = int(h * scale), int(w * scale)

        image_t = tf.convert_to_tensor(image, dtype=tf.float32)
        image_t = tf.expand_dims(image_t, axis=0)
        resized = tf.image.resize(image_t, [h_new, w_new], method='bicubic')
        scaled = tf.clip_by_value(resized / 255.0, 0.0, 1.0)

        return scaled

    def load_model(self):
        """
        Creates the model used to calculate cost using VGG19 as a base.

        The model outputs a list of feature maps from the layers defined in
        `style_layers` followed by the `content_layer`.

        Note: We use the pre-trained ImageNet weights and set the model to
        not be trainable, as we only need it for feature extraction.

        Returns:
            tf.keras.Model: The multi-output model.
        """
        # Load pre-trained VGG19 model without the classification head
        vgg = tf.keras.applications.VGG19(
                include_top=False, weights='imagenet')

        # Freeze the base model
        vgg.trainable = False

        # Extract the outputs of the specified layers
        style_outputs = [
                vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        # Combine outputs into a single list
        model_outputs = style_outputs + [content_output]

        # Construct the final Keras model
        model = tf.keras.Model(inputs=vgg.input, outputs=model_outputs)

        return model

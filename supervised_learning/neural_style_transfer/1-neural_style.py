#!/usr/bin/env python3
"""
Neural Style Transfer Module
"""

import numpy as np
import tensorflow as tf


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
        Creates the model used to calculate cost.
        """
        vgg = tf.keras.applications.VGG19(
                include_top=False, weights='imagenet')

        # In many NST implementations, you must apply VGG19 preprocessing.
        # Since our images are scaled [0, 1],
        # we need to scale them back to [0, 255]
        # and apply the VGG19 specific mean subtraction
        # if required by the test.

        x = vgg.input
        # Adding a preprocessing layer is
        # a clean way to handle the [0, 1] to VGG range
        model_input = tf.keras.applications.vgg19.preprocess_input(x * 255)

        # Re-initialize VGG with the processed input
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet',
                                          input_tensor=model_input)

        vgg.trainable = False

        style_outputs = [
                vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        model = tf.keras.Model(vgg.input, style_outputs + [content_output])

        return model

#!/usr/bin/env python3

'''
This module contains `projection_block` function.
'''

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Build a projection block as described in Deep Residual Learning
    for Image Recognition (2015).

    Args:
        A_prev: tensor, output from the previous layer
        filters: tuple or list containing f11, f3, f12
            f11: number of filters in the first 1x1 convolution
            f3: number of filters in the 3x3 convolution
            f12: number of filters in the second 1x1 convolution and shortcut
        s: stride of the first convolution in main and shortcut paths

    Returns:
        The activated output of the projection block
    """
    f11, f3, f12 = filters

    # He Normal initializer with seed=0
    initializer = K.initializers.HeNormal(seed=0)

    # Save the input value to apply the shortcut convolution later
    shortcut = A_prev

    # MAIN PATH
    # First component of main path (1x1 convolution with stride=s)
    x = K.layers.Conv2D(
        filters=f11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='valid',
        kernel_initializer=initializer
    )(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    # Second component of main path (3x3 convolution)
    x = K.layers.Conv2D(
        filters=f3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    # Third component of main path (1x1 convolution)
    x = K.layers.Conv2D(
        filters=f12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=initializer
    )(x)
    x = K.layers.BatchNormalization(axis=3)(x)

    # SHORTCUT PATH
    # 1x1 convolution with stride=s to match dimensions of main path
    shortcut = K.layers.Conv2D(
        filters=f12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='valid',
        kernel_initializer=initializer
    )(shortcut)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Final step: Add shortcut value to main path, THEN apply ReLU
    x = K.layers.Add()([x, shortcut])
    x = K.layers.Activation('relu')(x)

    return x

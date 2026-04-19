#!/usr/bin/env python3

'''
This module contains `identity_block` function.
'''

from tensorflow import keras


def identity_block(A_prev, filters):
    """
    Build an identity block as described in Deep Residual Learning 
    for Image Recognition (2015).
    
    Args:
        A_prev: tensor, output from the previous layer
        filters: tuple or list containing F11, F3, F12
            F11: number of filters in the first 1x1 convolution
            F3: number of filters in the 3x3 convolution
            F12: number of filters in the second 1x1 convolution
            
    Returns:
        The activated output of the identity block
    """
    f11, f3, f12 = filters
    
    # He Normal initializer with seed=0
    initializer = keras.initializers.HeNormal(seed=0)
    
    # Save the input value to add later to the main path
    shortcut = A_prev
    
    # First component of main path (1x1 convolution)
    x = keras.layers.Conv2D(
        filters=f11,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=initializer
    )(A_prev)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.Activation('relu')(x)
    
    # Second component of main path (3x3 convolution)
    x = keras.layers.Conv2D(
        filters=f3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.Activation('relu')(x)
    
    # Third component of main path (1x1 convolution)
    x = keras.layers.Conv2D(
        filters=f12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=initializer
    )(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation('relu')(x)

    return x

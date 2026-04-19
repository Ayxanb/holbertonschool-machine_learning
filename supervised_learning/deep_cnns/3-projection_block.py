#!/usr/bin/env python3
"""
Projection Block module for ResNet-50.
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Build a projection block as described in Deep Residual Learning
    for Image Recognition (2015).
    """
    f11, f3, f12 = filters
    init = K.initializers.HeNormal(seed=0)
    shortcut = A_prev

    # MAIN PATH
    x = K.layers.Conv2D(
        filters=f11, kernel_size=(1, 1), strides=(s, s),
        padding='valid', kernel_initializer=init
    )(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.ReLU()(x)

    x = K.layers.Conv2D(
        filters=f3, kernel_size=(3, 3), strides=(1, 1),
        padding='same', kernel_initializer=init
    )(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.ReLU()(x)

    x = K.layers.Conv2D(
        filters=f12, kernel_size=(1, 1), strides=(1, 1),
        padding='valid', kernel_initializer=init
    )(x)
    x = K.layers.BatchNormalization(axis=3)(x)

    # SHORTCUT PATH
    shortcut = K.layers.Conv2D(
        filters=f12, kernel_size=(1, 1), strides=(s, s),
        padding='valid', kernel_initializer=init
    )(shortcut)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Addition and final ReLU
    x = K.layers.Add()([x, shortcut])
    x = K.layers.ReLU()(x)

    return x

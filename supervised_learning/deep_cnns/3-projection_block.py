#!/usr/bin/env python3
"""
ResNet-50 Architecture module.
"""
from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Build the ResNet-50 architecture as described in Deep Residual
    Learning for Image Recognition (2015).

    Returns:
        The Keras Model instance for ResNet-50.
    """
    inputs = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.HeNormal(seed=0)

    # Stage 1
    x = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(inputs)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.ReLU()(x)

    # Max pooling downsamples to 56x56
    x = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(x)

    # Stage 2 (3 blocks)
    x = projection_block(x, filters=(64, 64, 256), s=1)
    x = identity_block(x, filters=(64, 64, 256))
    x = identity_block(x, filters=(64, 64, 256))

    # Stage 3 (4 blocks)
    x = projection_block(x, filters=(128, 128, 512), s=2)
    x = identity_block(x, filters=(128, 128, 512))
    x = identity_block(x, filters=(128, 128, 512))
    x = identity_block(x, filters=(128, 128, 512))

    # Stage 4 (6 blocks)
    x = projection_block(x, filters=(256, 256, 1024), s=2)
    x = identity_block(x, filters=(256, 256, 1024))
    x = identity_block(x, filters=(256, 256, 1024))
    x = identity_block(x, filters=(256, 256, 1024))
    x = identity_block(x, filters=(256, 256, 1024))
    x = identity_block(x, filters=(256, 256, 1024))

    # Stage 5 (3 blocks)
    x = projection_block(x, filters=(512, 512, 2048), s=2)
    x = identity_block(x, filters=(512, 512, 2048))
    x = identity_block(x, filters=(512, 512, 2048))

    # Average Pooling and Classification Head
    x = K.layers.AveragePooling2D(
        pool_size=(7, 7), padding='valid'
    )(x)

    x = K.layers.Flatten()(x)

    outputs = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(x)

    # Instantiate the model (using default name 'model' for checker)
    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model

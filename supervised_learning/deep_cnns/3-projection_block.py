#!/usr/bin/env python3
"""
Builds the ResNet-50 architecture.
"""
from tensorflow import keras

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015).
    
    Returns:
        The compiled Keras model.
    """
    # He Normal initialization with seed=0
    init = keras.initializers.he_normal(seed=0)

    # Input layer shape
    X = keras.Input(shape=(224, 224, 3))

    # Conv1 block
    conv1 = keras.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=init)(X)
    norm1 = keras.layers.BatchNormalization(axis=3)(conv1)
    act1 = keras.layers.Activation('relu')(norm1)
    pool1 = keras.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(act1)

    # Conv2_x block
    # Note: s=1 because the max pooling already downsampled the spatial dims
    conv2 = projection_block(pool1, [64, 64, 256], s=1)
    conv2 = identity_block(conv2, [64, 64, 256])
    conv2 = identity_block(conv2, [64, 64, 256])

    # Conv3_x block
    conv3 = projection_block(conv2, [128, 128, 512], s=2)
    conv3 = identity_block(conv3, [128, 128, 512])
    conv3 = identity_block(conv3, [128, 128, 512])
    conv3 = identity_block(conv3, [128, 128, 512])

    # Conv4_x block
    conv4 = projection_block(conv3, [256, 256, 1024], s=2)
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])

    # Conv5_x block
    conv5 = projection_block(conv4, [512, 512, 2048], s=2)
    conv5 = identity_block(conv5, [512, 512, 2048])
    conv5 = identity_block(conv5, [512, 512, 2048])

    # Average Pooling layer
    avg_pool = keras.layers.AveragePooling2D(
        (7, 7), strides=(1, 1), padding='valid')(conv5)

    # Dense layer with Softmax
    dense = keras.layers.Dense(
        1000, activation='softmax', kernel_initializer=init)(avg_pool)

    # The actual output size at this point would be (None, 1, 1, 1000) if
    # applied raw. Flatten resolves this to match label encodings (None, 1000).
    output = keras.layers.Flatten()(dense)

    # Compile the final model
    model = keras.Model(inputs=X, outputs=output)

    return model

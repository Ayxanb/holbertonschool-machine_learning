#!/usr/bin/env python3
"""
Module to build a modified LeNet-5 architecture using Keras.
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture.

    Args:
        X: K.Input of shape (m, 28, 28, 1) containing the input images.

    Returns:
        A K.Model compiled with Adam optimizer and accuracy metrics.
    """
    # Define the initializer for reproducibility
    init = K.initializers.HeNormal(seed=0)

    # Layer 1: Conv 6 kernels 5x5, same padding, ReLU
    layer1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    # Layer 2: Max Pooling 2x2, stride 2x2
    layer2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(layer1)

    # Layer 3: Conv 16 kernels 5x5, valid padding, ReLU
    layer3 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(layer2)

    # Layer 4: Max Pooling 2x2, stride 2x2
    layer4 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(layer3)

    # Flatten the output for Dense layers
    flatten = K.layers.Flatten()(layer4)

    # Layer 5: Fully connected 120 nodes, ReLU
    layer5 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(flatten)

    # Layer 6: Fully connected 84 nodes, ReLU
    layer6 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(layer5)

    # Layer 7: Output layer 10 nodes, Softmax
    outputs = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
    )(layer6)

    # Create and compile the model
    model = K.Model(inputs=X, outputs=outputs)
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

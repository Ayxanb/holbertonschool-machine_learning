#!/usr/bin/env python3
"""
This module contains a function to train a Keras model
with validation, early stopping, learning rate decay,
and saving the best iteration.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and various callbacks.

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes)
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data
        validation_data: data to validate the model with, if not None
        early_stopping: boolean indicating whether to use early stopping
        patience: the patience used for early stopping
        learning_rate_decay: boolean indicating whether to use LR decay
        alpha: the initial learning rate
        decay_rate: the decay rate for inverse time decay
        save_best: boolean indicating whether to save the best model
        filepath: file path where the model should be saved
        verbose: boolean determining if output should be printed
        shuffle: boolean determining whether to shuffle batches every epoch

    Returns:
        The History object generated after training the model.
    """
    callbacks = []

    if validation_data:
        # Early Stopping
        if early_stopping:
            es_callback = K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
            callbacks.append(es_callback)

        # Learning Rate Decay
        if learning_rate_decay:
            def scheduler(epoch):
                # Calculates the learning rate based on inverse time decay
                return alpha / (1 + decay_rate * epoch)

            lr_callback = K.callbacks.LearningRateScheduler(
                schedule=scheduler,
                verbose=1
            )
            callbacks.append(lr_callback)

        # Save Best Model
        if save_best and filepath:
            save_callback = K.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
            callbacks.append(save_callback)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )

    return history

#!/usr/bin/env python3
"""
Train a CNN to classify CIFAR-10 dataset using Transfer Learning.
"""

import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model.
    """
    X_p = K.applications.densenet.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    # Load the CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Preprocess the data
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Build the base model and feature extractor
    inputs = K.Input(shape=(32, 32, 3))

    # Lambda layer to scale up the data to the correct size
    resize = K.layers.Lambda(lambda x: tf.image.resize(x, (160, 160)))
    resized_inputs = resize(inputs)

    # Pre-trained base application
    base_model = K.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    # Model to extract features
    features = base_model(resized_inputs)
    feature_extractor = K.Model(inputs=inputs, outputs=features)

    # Hint 3: Compute the output of the frozen layers ONCE
    print("Pre-computing training features...")
    train_features = feature_extractor.predict(X_train, batch_size=128)

    print("Pre-computing testing features...")
    test_features = feature_extractor.predict(X_test, batch_size=128)

    # Build the classifier head
    feat_dim = base_model.output_shape[1]
    head_inputs = K.Input(shape=(feat_dim,))
    x = K.layers.Dense(512, activation='relu')(head_inputs)
    x = K.layers.Dropout(0.3)(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.2)(x)
    head_outputs = K.layers.Dense(10, activation='softmax')(x)

    classifier = K.Model(inputs=head_inputs, outputs=head_outputs)

    classifier.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the classifier on the pre-computed features
    classifier.fit(
        train_features, Y_train,
        validation_data=(test_features, Y_test),
        epochs=10,
        batch_size=128
    )

    # Combine the lambda layer, base model, and classifier into one model
    final_outputs = classifier(base_model(resize(inputs)))
    final_model = K.Model(inputs=inputs, outputs=final_outputs)

    # Compile the saved model
    final_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Save the compiled model
    final_model.save('cifar10.h5')

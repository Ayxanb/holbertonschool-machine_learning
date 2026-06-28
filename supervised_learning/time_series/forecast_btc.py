# forecast_btc.py
import tensorflow as tf
import numpy as np


def create_dataset(data, window_size=24, batch_size=64, is_training=True):
    """Creates a tf.data.Dataset windowed pipeline for the RNN model."""
    dataset = tf.data.Dataset.from_tensor_slices(data)

    # Create windows of size window_size + 1 (24 inputs + 1 target)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size + 1))

    # Split window into features (past 24h) and target (next 1h Close price)
    # Target index 0 corresponds to the 'Close' column in our preprocessed data
    dataset = dataset.map(lambda w: (w[:-1, :], w[-1, 0]))

    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_rnn_model(input_shape):
    """Builds a Keras sequential RNN model using GRU layers."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.GRU(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ]
    )
    return model


def main():
    # Load preprocessed arrays
    loaded = np.load("preprocessed_btc.npz")
    data = loaded["data"]

    # Split into train and validation sets (80% / 20%)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Prepare tf.data Datasets
    window_size = 24
    batch_size = 128

    train_dataset = create_dataset(
        train_data, window_size, batch_size, is_training=True
    )
    val_dataset = create_dataset(
        val_data, window_size, batch_size, is_training=False
    )

    # Instantiate and compile model
    input_shape = (window_size, data.shape[1])
    model = build_rnn_model(input_shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    model.summary()

    # Define training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]

    # Train model
    model.fit(
        train_dataset,
        epochs=20,
        validation_data=val_dataset,
        callbacks=callbacks
    )


if __name__ == "__main__":
    main()

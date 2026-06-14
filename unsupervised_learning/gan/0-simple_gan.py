#!/usr/bin/env python3

'''
This module contains `Simple_GAN` class
'''

import tensorflow as tf


class Simple_GAN(tf.keras.Model):
    """A Simple GAN class compliant with Keras lifecycle and testing."""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, learning_rate=0.001, **kwargs):
        """Initializes the GAN components and hyper-parameters."""
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.learning_rate = learning_rate

        # Placeholders for optimization objects built during compile
        self.g_optimizer = None
        self.d_optimizer = None
        self.loss_fn = None

    def compile(self):
        """Configures the model components and signals Keras compilation.

        Calls super().compile() to properly flip internal Keras flags
        so that lifecycle methods like `.fit()` function correctly.
        """
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # Satisfy internal Keras compilation verification flags
        super().compile(optimizer=self.d_optimizer, loss=self.loss_fn)

    def train_step(self, data):
        """Performs a single training step.

        Keras `.fit()` automatically calls this method and passes only
        the current batch of data (`real_images`).

        Args:
            data: A batch of true data tensors from the dataset.

        Returns:
            A dictionary containing the loss values for monitoring.
        """
        real_images = data
        batch_size = tf.shape(real_images)[0]

        # Targets for Binary Crossentropy
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        # Sample random noise using stored instance attribute latent_dim
        z = self.latent_generator(batch_size, self.latent_dim)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake samples
            fake_images = self.generator(z, training=True)

            # Evaluate both sets with Discriminator
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            # Compute losses
            d_loss_real = self.loss_fn(real_labels, real_output)
            d_loss_fake = self.loss_fn(fake_labels, fake_output)
            d_loss = d_loss_real + d_loss_fake

            g_loss = self.loss_fn(real_labels, fake_output)

        # Extract and apply gradients
        gradients_of_discriminator = disc_tape.gradient(
            d_loss, self.discriminator.trainable_variables
        )
        gradients_of_generator = gen_tape.gradient(
            g_loss, self.generator.trainable_variables
        )

        self.d_optimizer.apply_gradients(
            zip(gradients_of_discriminator,
                self.discriminator.trainable_variables)
        )
        self.g_optimizer.apply_gradients(
            zip(gradients_of_generator,
                self.generator.trainable_variables)
        )

        # Keras expects a dictionary of metric/loss tracking names back
        return {"d_loss": d_loss, "g_loss": g_loss}

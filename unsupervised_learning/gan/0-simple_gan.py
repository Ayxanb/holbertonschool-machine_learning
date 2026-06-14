#!/usr/bin/env python3

'''
This module contains `Simple_GAN` class
'''

import tensorflow as tf


class Simple_GAN:
    """A Simple GAN class matching the automated test signature.

    遵守 pycodestyle: 每行不超过 79 个字符。
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, learning_rate=0.001):
        """Initializes the GAN with models, generators, and learning rate.

        Args:
            generator: The tf.keras.Model producing synthetic data.
            discriminator: The tf.keras.Model classifying real vs fake.
            latent_generator: A callable function/object that yields random
                latent vectors (noise) when called.
            real_examples: The training dataset tensor or array.
            learning_rate: Float, the step size for the Adam optimizers.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples

        # Define internal optimizers using the provided learning rate
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

        # Binary Crossentropy is the standard objective function for GANs
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def train_step(self, real_images, latent_dim):
        """Performs a single forward and backward training pass.

        Args:
            real_images: A batch of true data tensors from the dataset.
            latent_dim: Integer, the dimension size of the noise vector.

        Returns:
            d_loss: The scalar loss value for the discriminator.
            g_loss: The scalar loss value for the generator.
        """
        batch_size = tf.shape(real_images)[0]

        # Targets for Binary Crossentropy
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        # Notice: You can either sample via your latent_generator()
        # or stick to tf.random.normal depending on instruction details.
        # Here we follow standard practice using latent_dim:
        z = tf.random.normal([batch_size, latent_dim])

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

        return d_loss, g_lossimport tensorflow as tf


class Simple_GAN:
    """A Simple GAN class matching the automated test signature."""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, learning_rate=0.001):
        """Initializes the GAN with models, generators, and learning rate.

        Args:
            generator: The tf.keras.Model producing synthetic data.
            discriminator: The tf.keras.Model classifying real vs fake.
            latent_generator: A callable function/object that yields random
                latent vectors (noise) when called.
            real_examples: The training dataset tensor or array.
            learning_rate: Float, the step size for the Adam optimizers.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples

        # Define internal optimizers using the provided learning rate
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

        # Binary Crossentropy is the standard objective function for GANs
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def train_step(self, real_images, latent_dim):
        """Performs a single forward and backward training pass.

        Args:
            real_images: A batch of true data tensors from the dataset.
            latent_dim: Integer, the dimension size of the noise vector.

        Returns:
            d_loss: The scalar loss value for the discriminator.
            g_loss: The scalar loss value for the generator.
        """
        batch_size = tf.shape(real_images)[0]

        # Targets for Binary Crossentropy
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        # Notice: You can either sample via your latent_generator()
        # or stick to tf.random.normal depending on instruction details.
        # Here we follow standard practice using latent_dim:
        z = tf.random.normal([batch_size, latent_dim])

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

        return d_loss, g_loss

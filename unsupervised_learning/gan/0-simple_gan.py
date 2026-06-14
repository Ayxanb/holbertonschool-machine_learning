#!/usr/bin/env python3

'''
This module contains `Simple_GAN` class
'''

import tensorflow as tf


class Simple_GAN:
    """A clean, pycodestyle-compliant implementation of a Simple GAN.

    This class manages the training step for both the Generator and
    the Discriminator networks using TensorFlow's GradientTape.
    """

    def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
                 loss_fn):
        """Initializes the GAN with models, optimizers, and a loss function.

        Args:
            generator: The tf.keras.Model producing synthetic data.
            discriminator: The tf.keras.Model classifying real vs fake.
            g_optimizer: tf.keras.optimizers.Optimizer for the generator.
            d_optimizer: tf.keras.optimizers.Optimizer for the discriminator.
            loss_fn: tf.keras.losses.Loss instance (typically BCE).
        """
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images, latent_dim):
        """Performs a single forward and backward pass for both networks.

        Args:
            real_images: A batch of true data tensors from the dataset.
            latent_dim: Integer, the size of the input noise vector.

        Returns:
            d_loss: The scalar loss value for the discriminator.
            g_loss: The scalar loss value for the generator.
        """
        batch_size = tf.shape(real_images)[0]

        # Ground truth labels for the loss functions
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        # Sample random noise from a standard normal distribution
        z = tf.random.normal([batch_size, latent_dim])

        # Track operations to compute gradients
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate a batch of fake images
            fake_images = self.generator(z, training=True)

            # Pass both real and fake images through the discriminator
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            # Discriminator Loss: minimize error on both real and fake batches
            d_loss_real = self.loss_fn(real_labels, real_output)
            d_loss_fake = self.loss_fn(fake_labels, fake_output)
            d_loss = d_loss_real + d_loss_fake

            # Generator Loss: maximize the probability of D being fooled
            g_loss = self.loss_fn(real_labels, fake_output)

        # Calculate gradients with respect to respective trainable weights
        gradients_of_discriminator = disc_tape.gradient(
            d_loss, self.discriminator.trainable_variables
        )
        gradients_of_generator = gen_tape.gradient(
            g_loss, self.generator.trainable_variables
        )

        # Apply gradients to optimize the weights
        self.d_optimizer.apply_gradients(
            zip(gradients_of_discriminator,
                self.discriminator.trainable_variables)
        )
        self.g_optimizer.apply_gradients(
            zip(gradients_of_generator,
                self.generator.trainable_variables)
        )

        return d_loss, g_loss

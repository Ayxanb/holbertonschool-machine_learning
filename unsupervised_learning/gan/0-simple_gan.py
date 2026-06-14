#!/usr/bin/env python3

'''
This module contains `Simple_GAN` class
'''

import tensorflow as tf


class Simple_GAN(tf.keras.Model):
    """A Simple GAN class compliant with Keras and the testing framework. """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, learning_rate=0.001, **kwargs):
        """Initializes the GAN components and sets up learning rates."""
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
        """Configures the model components for training."""
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )
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

        # Sample random noise using the provided latent_generator function
        z = self.latent_generator(batch_size, latent_dim)

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

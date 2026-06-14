#!/usr/bin/env python3
import torch


class Simple_GAN:

    def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
                 loss_fn):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images, latent_dim):
        """Performs a single training step for both the D and G networks."""
        batch_size = real_images.size(0)
        device = real_images.device

        # Labels for loss calculation
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # 1. Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        self.d_optimizer.zero_grad()

        # Test Discriminator on real images
        outputs_real = self.discriminator(real_images)
        d_loss_real = self.loss_fn(outputs_real, real_labels)

        # Generate fake images from random latent vectors
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_images = self.generator(z)

        # Test Discriminator on fake images
        # We detach fake_images because we are only training D here
        outputs_fake = self.discriminator(fake_images.detach())
        d_loss_fake = self.loss_fn(outputs_fake, fake_labels)

        # Combine losses and update Discriminator
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        # 2. Train Generator: maximize log(D(G(z)))
        self.g_optimizer.zero_grad()

        # We want the Discriminator to mistake these fakes for real
        outputs_fake_for_g = self.discriminator(fake_images)
        g_loss = self.loss_fn(outputs_fake_for_g, real_labels)

        # Update Generator
        g_loss.backward()
        self.g_optimizer.step()

        return d_loss.item(), g_loss.item()

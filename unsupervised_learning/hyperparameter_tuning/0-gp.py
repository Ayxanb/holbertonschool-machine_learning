#!/usr/bin/env python3
"""
This module provides the GaussianProcess class for performing
1D Gaussian process regression.
"""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initializes the Gaussian Process.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix using the RBF kernel.
        """
        # Calculate squared Euclidean distance:
        # dist = (x - y)^2 = x^2 + y^2 - 2xy
        sqdist = (
            np.sum(X1**2, 1).reshape(-1, 1)
            + np.sum(X2**2, 1)
            - 2 * np.dot(X1, X2.T)
        )

        # RBF Kernel formula: K(x, y) = sigma_f^2 * exp(-0.5 * dist / l^2)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

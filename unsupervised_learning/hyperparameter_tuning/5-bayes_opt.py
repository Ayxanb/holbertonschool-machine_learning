#!/usr/bin/env python3
"""
This module provides the BayesianOptimization class for performing
Bayesian optimization on a noiseless 1D Gaussian process.
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initializes the Bayesian Optimization object.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement.
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi

        Z = np.zeros(sigma.shape)
        mask = sigma > 0
        Z[mask] = imp[mask] / sigma[mask]
        Z[~mask] = 0

        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0] = 0

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function.
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Check if point has already been sampled
            if X_next in self.gp.X:
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt

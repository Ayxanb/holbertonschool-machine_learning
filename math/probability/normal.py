#!/usr/bin/env python3
"""
This module contains `Normal` class
"""

pi = 3.1415926536
e = 2.7182818285


class Normal:
    """
    Normal distribution class
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the Normal class
        data: List of the data to be used to estimate the distribution
        mean: The mean of the distribution
        stddev: The standard deviation of the distribution
        """

        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')

            self.mean = float(mean)
            self.stddev = float(stddev)

        elif not isinstance(data, list):
            raise TypeError('data must be a list')

        elif len(data) < 2:
            raise ValueError('data must contain multiple values')

        else:
            n = len(data)
            self.mean = sum(data) / n
            _sum = sum((x - self.mean) ** 2 for x in data)
            self.stddev = (_sum / n) ** 0.5

    def z_score(self, x):
        """
        The `z` Value (The Standardized Score)
        """

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        The `x` Value (The Raw Score)
        """

        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculates the PDF
        """
        return (
            (1 / (self.stddev * (2 * pi) ** 0.5)) *
            e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
        )

    def erf(self, x):
        """
        Manual implementation of the Error Function (erf)
        Approximation via Abramowitz and Stegun
        """
        # Save the sign of x
        sign = 1 if x >= 0 else -1
        x = abs(x)

        # Constants for the approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (
            ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * e**(-x*x)

        return sign * y

    def cdf(self, x):
        """
        Calculates the CDF
        """
        return 0.5 * (
            1 + self.erf((x - self.mean) / (2 ** 0.5 * self.stddev))
        )

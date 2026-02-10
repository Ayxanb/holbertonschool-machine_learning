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

    def cdf(self, x):
        """
        Calculates the CDF
        """
        val = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = (2 / (pi ** 0.5)) * (
            val - (val ** 3) / 3 + (val ** 5) / 10 -
            (val ** 7) / 42 + (val ** 9) / 216
        )
        return 0.5 * (1 + erf)

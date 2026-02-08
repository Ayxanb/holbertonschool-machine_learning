#!/usr/bin/env python3
"""
This module defines a Poisson distribution class.
"""
import math


class Poisson:
    """
    Represents a Poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes a Poisson distribution.
        """

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        elif not isinstance(data, list):
            raise TypeError("data must be a list")

        elif len(data) < 2:
            raise ValueError("data must contain multiple values")

        else:
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        '''
        Calculates the value of the PMF for a given number of "successes"
        '''

        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        return ((self.lambtha ** k) * math.exp(-self.lambtha) /
                math.factorial(k))

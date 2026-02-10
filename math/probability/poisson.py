#!/usr/bin/env python3
"""
This module defines a Poisson distribution class.
"""

pi = 3.1415926536
e = 2.7182818285


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
        Probability Mass Function
        The PMF tells you the exact probability
        that an event happens exactly `k` times.
        '''

        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        return (e ** (-self.lambtha)) * (self.lambtha ** k) / factorial


    def cdf(self, k):
        '''
        Cumulative Distribution Function
        The CDF tells the probability that
        the number of events is less than or equal to a certain value.
        It's calculated by summing the PMFs for all values from 0 up to `k`
        '''

        _cdf = 0
        for i in range(k):
            _cdf += self.pmf(i)

        return _cdf

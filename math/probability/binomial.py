#!/usr/bin/env python3
'''
This module contains `Binomial` class
'''


class Binomial:
    '''
    Binomial distribution class
    '''

    def __init__(self, data=None, n=1, p=0.5):
        '''
        Initializes Binomial class
        data: List of the data to be used to estimate the distribution
        n: The number of Bernoulli trials
        p: The probability of a "success"
        '''

        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')

            if not (0 < p < 1):
                raise ValueError('p must be greater than 0 and less than 1')

            self.n = int(n)
            self.p = float(p)

        elif not isinstance(data, list):
            raise TypeError("data must be a list")

        elif len(data) < 2:
            raise ValueError("data must contain multiple values")

        else:
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p_initial = 1 - (variance / mean)
            self.n = int(round(mean / p_initial))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """ Calculates the PMF for exactly k successes """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        def fact(num):
            f = 1
            for i in range(1, num + 1):
                f *= i
            return f

        n_choose_k = fact(self.n) / (fact(k) * fact(self.n - k))
        return n_choose_k * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """ Calculates the CDF for at most k successes """
        k = int(k)
        if k < 0:
            return 0
        if k >= self.n:
            return 1

        total_p = 0
        for i in range(k + 1):
            total_p += self.pmf(i)

        return total_p

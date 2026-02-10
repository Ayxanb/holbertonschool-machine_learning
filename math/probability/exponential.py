#!/usr/bin/env python3
'''
This module contains `Exponential` class
'''

e = 2.7182818285


class Exponential:
    '''
    Exponential class
    '''

    def __init__(self, data=None, lambtha=1.):
        '''
        Initializes Exponential class
        data: List of the data to be used to estimate the distribution
        lambtha: The expected number of occurences in a given time frame
        '''

        if data is None:
            if lambtha <= 0.:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = lambtha

        elif not isinstance(data, list):
            raise TypeError('data must be a list')

        elif len(data) < 2:
            raise ValueError('data must contain multiple values')

        else:
            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        '''
        Probability Density Function
        The PDF describes the likelihood of the random variable `x`
        '''

        if x < 0:
            return 0

        return self.lambtha * e ** (-self.lambtha * x)

    def cdf(self, x):
        '''
        Cumulative Distribution Function
        The CDF calculates the probability
        that the time until the next event is less than or equal to `x`.
        '''

        if x < 0:
            return 0

        return 1 - e ** (-self.lambtha * x)

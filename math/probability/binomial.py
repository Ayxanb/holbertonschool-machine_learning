#!/usr/bin/env python3
'''
'''


class Binomial:
    '''
    '''

    def __init__(self, data=None, n=1, p=0.5):
        '''
        Initializes Binomial class
        data: List of the data to be used to estimate the distribution
        n: The number of Bernoulli trials
        p: The probability of a "success"
        '''

        if data is None:
            if n < 0:
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

#!/usr/bin/env python3
'''
This module contains `Normal` class
'''


class Normal:
    '''
    Normal distribution class
    '''

    def __init__(self, data=None, mean=0., stddev=1.):
        '''
        Initializes the Normal class
        data: List of the data to be used to estimate the distribution
        mean: The mean of the distribution
        stddev: The standard deviation of the distribution
        '''

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
            self.stddev = (_sum / (n - 1)) ** 0.5

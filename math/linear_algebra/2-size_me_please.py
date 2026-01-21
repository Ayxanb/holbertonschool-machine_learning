#!/usr/bin/env python3

'''
This module contains `matrix_shape`
'''


def matrix_shape(matrix):
    '''
    returns the shape of a matrix in a list
    '''

    shape = []

    def recursive(a):
        if isinstance(a, list):
            shape.append(len(a))
            recursive(a[0])

    recursive(matrix)
    return shape

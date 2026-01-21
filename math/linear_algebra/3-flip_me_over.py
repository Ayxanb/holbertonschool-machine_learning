#!/usr/bin/env python3

'''
This module contains `matrix_transpose` function
'''


def matrix_transpose(matrix):
    '''
    transposes a matrix
    '''

    return [list(row) for row in zip(*matrix)]

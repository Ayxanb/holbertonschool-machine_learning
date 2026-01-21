#!/usr/bin/env python3

'''
This module contains `add_matrices2D` function
'''


def add_matrices2D(mat1, mat2):
    '''
    adds two 2D matrices 
    '''

    matrix_shape = __import__('2-size_me_please')

    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    return [[a + b for a, b in zip(arr1, arr2)] for arr1, arr2 in zip(mat1, mat2)]

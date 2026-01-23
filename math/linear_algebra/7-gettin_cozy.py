#!/usr/bin/env python3

'''
This module contains `cat_matrices2D` function
'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''
    concatinates 2 matrices along with axis
    '''

    if axis == 0:
        if not mat1 or not mat2 or len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1 + mat2]

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [r1 + r2 for r1, r2 in zip(mat1, mat2)]

    return None

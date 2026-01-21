#!/usr/bin/env python3

'''
This module contains `add_matrices2D` function
'''


def add_matrices2D(mat1, mat2):
    '''
    adds two 2D matrices
    '''

    if len(mat1) == 0 or len(mat2) == 0:
        return None

    result = []
    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None
        result.append([mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))])

    return result

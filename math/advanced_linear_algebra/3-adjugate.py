#!/usr/bin/env python3
'''
This module contains `adjugate` function
'''

def adjugate(matrix):
    '''
    Calculates the adjugate matrix of a matrix
    '''
    cofactor = __import__('2-cofactor').cofactor

    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not matrix or not matrix[0]:
        raise ValueError("matrix must be a non-empty square matrix")

    size = len(matrix)

    if not all(len(row) == size for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if size == 1:
        return [[1]]

    cof_mat = cofactor(matrix)

    return [list(row) for row in zip(*cof_mat)]

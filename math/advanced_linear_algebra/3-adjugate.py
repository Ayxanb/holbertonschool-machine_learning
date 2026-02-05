#!/usr/bin/env python3

'''
This module contains `adjugate` function
'''


def adjugate(matrix):
    '''
    Returns the adjugate matrix
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

    cofactor_matrix = cofactor(matrix)
    return [list(i) for i in zip(*matrix)]

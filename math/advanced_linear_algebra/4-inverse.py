#!/usr/bin/env python3

'''
This module contains `inverse` function
'''


def inverse(matrix):
    '''
    returns the inverse of matrix
    '''

    determinant = __import__('0-determinant').determinant
    adjugate = __import__('3-adjugate').adjugate
    
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


    if determinant(matrix) == 0:
        return None

    adjugate_matrix = adjugate(matrix)
    det = determinant(matrix)

    return list(map(lambda x: x/det, row for row in adjugate_matrix))

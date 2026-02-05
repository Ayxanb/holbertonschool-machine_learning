#!/usr/bin/env python3
'''
This module contains `inverse` function
'''

def inverse(matrix):
    '''
    Calculates the inverse of a matrix
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

    det = determinant(matrix)
    if det == 0:
        return None

    adj_mat = adjugate(matrix)
    inv_matrix = [[val / det for val in row] for row in adj_mat]

    return inv_matrix

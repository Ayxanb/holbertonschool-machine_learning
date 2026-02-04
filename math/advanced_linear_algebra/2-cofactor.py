#!/usr/bin/env python3
"""
This module contains the cofactor function
"""


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix
    """
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

    determinant = __import__('0-determinant').determinant

    cofactor_matrix = []
    for i in range(size):
        row_cofactors = []
        for j in range(size):
            sub_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            
            minor_val = determinant(sub_matrix)
            cofactor_val = ((-1) ** (i + j)) * minor_val
            row_cofactors.append(cofactor_val)
            
        cofactor_matrix.append(row_cofactors)

    return cofactor_matrix

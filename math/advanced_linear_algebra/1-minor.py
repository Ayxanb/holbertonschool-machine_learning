#!/usr/bin/env python3
'''
This module contains `minor` function
'''


def minor(matrix):
    """
    Calculates the minor matrix of a square matrix.
    """

    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")


    if matrix == [] or matrix == [[]]:
        raise ValueError('matrix must be a non-empty square matrix')

    rows = len(matrix)

    if not all(len(row) == rows for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix"y 
    if rows == 1:
        return [[1]]

    determinant = __import__('0-determinant').determinant
    minor_matrix = []

    for i in range(rows):
        current_row = []
        for j in range(rows):
            sub_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            current_row.append(determinant(sub_matrix))
        minor_matrix.append(current_row)

    return minor_matrix

#!/usr/bin/env python3

'''
This module contains `determinant` function
'''


def determinant(matrix):
    """
    Calculates the determinant of a square matrix.
    """

    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]] or matrix == []:
        return 1

    rows = len(matrix)

    if not all(len(row) == rows for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if rows == 1:
        return matrix[0][0]

    elif rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(rows):
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(minor)

    return det

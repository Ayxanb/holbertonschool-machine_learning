#!/usr/bin/env python3
"""
This module contains the minor function
"""


def minor(matrix):
    """
    Calculates the minor matrix of a matrix
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

    minor_matrix = []
    for i in range(size):
        row_minors = []
        for j in range(size):
            sub_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            row_minors.append(determinant(sub_matrix))
        minor_matrix.append(row_minors)

    return minor_matrix

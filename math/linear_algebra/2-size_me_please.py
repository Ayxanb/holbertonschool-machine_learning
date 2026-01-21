#!/usr/bin/env python3

def matrix_shape(matrix):
    shape = []

    def recursive(a):
        if isinstance(a, list):
            shape.append(len(a))
            recursive(a[0])

    recursive(matrix)
    return shape

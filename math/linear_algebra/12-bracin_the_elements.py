#!/usr/bin/env python3
'''
This module contains `np_elementwise`
'''


def np_elementwise(mat1, mat2):
    """
    Returns a tuple of element-wise sum, difference, product, and quotient.
    """

    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)

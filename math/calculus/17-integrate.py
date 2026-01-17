#!/usr/bin/env python3
"""Polynomial integral calculation."""


def poly_integral(poly, C=0):
    """Return the integral of a polynomial as a list."""

    if not isinstance(poly, list) or not poly:
        return None

    if not isinstance(C, int):
        return None

    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    result = [C]

    for i, coef in enumerate(poly):
        value = coef / (i + 1)
        if value.is_integer():
            value = int(value)
        result.append(value)

    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result

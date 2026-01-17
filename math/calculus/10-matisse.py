#!/usr/bin/env python3
"""Polynomial derivative calculation."""


def poly_derivative(poly):
    """Return the derivative of a polynomial as a list."""

    if not isinstance(poly, list) or not poly:
        return None

    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    if len(poly) == 1:
        return [0]

    derivative = [poly[i] * i for i in range(1, len(poly))]

    if all(coef == 0 for coef in derivative):
        return [0]

    return derivative

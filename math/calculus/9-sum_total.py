#!/usr/bin/env python3
"""
This module provides a utility for calculating mathematical series.
Specifically, it focuses on the sum of the first n squares.
"""

def summation_i_squared(n):
    """
    Calculates the sum of squares from 1 to n using the closed-form formula.
    Formula used: n(n + 1)(2n + 1) / 6
    """
    return n * (n + 1) * (2 * n + 1) / 6

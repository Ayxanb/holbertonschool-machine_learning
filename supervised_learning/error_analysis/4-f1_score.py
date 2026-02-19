#!/usr/bin/env python3

'''
This module contains `f1_score` function.
'''

import numpy as np


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.
    """

    sensitivity_func = __import__('1-sensitivity').sensitivity
    precision_func = __import__('2-precision').precision

    s = sensitivity_func(confusion)
    p = precision_func(confusion)

    f1 = 2 * (p * s) / (p + s)
    return f1

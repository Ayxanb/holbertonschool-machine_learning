#!/usr/bin/env python3
'''
This module contains `sensitivity` function.
'''

import numpy as np


def sensitivity(confusion):
    '''
    Calculates the sensitivity for each class in a confusion matrix.
    '''

    # 1. Extract the True Positives
    true_positives = np.diag(confusion)

    # 2. Calculate the total Actual Positives for each class
    # This is the sum of each row
    actual_positives = np.sum(confusion, axis=1)

    # 3. Divide TP by Actual Positives to get Sensitivity
    sensitivity_array = true_positives / actual_positives

    return sensitivity_array

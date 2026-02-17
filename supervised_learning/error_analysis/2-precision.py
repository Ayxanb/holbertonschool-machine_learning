#!/usr/bin/env python3
'''
This module contains `precision` function.
'''

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.
    """

    # 1. Extract the True Positives (correct guesses for each class).
    true_positives = np.diag(confusion)
    
    # 2. The total Predicted Positives for each class (sum of each column)
    predicted_positives = np.sum(confusion, axis=0)
    
    # 3. Divide TP by Predicted Positives (precision for every class)
    precision_array = true_positives / predicted_positives
    
    return precision_array

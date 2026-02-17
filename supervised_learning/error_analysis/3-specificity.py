#!/usr/bin/env python3
'''
This module contains `specificity` function.
'''

import numpy as np


def specificity(confusion):
    '''
    Calculates the specificity for each class in a confusion matrix.
    '''

    # 1. Total number of all samples in the matrix
    total_samples = np.sum(confusion)

    # 2. Sum of each row (Actual Positives for each class)
    row_sums = np.sum(confusion, axis=1)

    # 3. Sum of each column (Predicted Positives for each class)
    col_sums = np.sum(confusion, axis=0)

    # 4. True Positives
    tp = np.diag(confusion)

    # 5. Actual Negatives = Total - Row Sum
    actual_negatives = total_samples - row_sums

    # True Negatives = Total - Row Sum - Column Sum + True Positive
    tn = total_samples - (row_sums + col_sums) + tp

    # 6. Specificity = TN / Actual Negatives
    return tn / actual_negatives

#!/usr/bin/env python3

'''
This module provides `analyze` function
'''


def analyze(df):
    '''
    takes a pd.DataFrame and:

    Computes descriptive statistics for all columns except the Timestamp column.
    Returns a new pd.DataFrame containing these statistics.
    '''

    return df.drop(columns=["Timestamp"]).describe()

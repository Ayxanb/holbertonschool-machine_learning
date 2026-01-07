#!/usr/bin/env python3

'''
This module provides `high` function
'''


def high(df):
    '''
    takes a pd.DataFrame and:

    Sorts it by the High price in descending order.
    Returns: the sorted pd.DataFrame.
    '''

    return df.sort_values(by='High', ascending=False)

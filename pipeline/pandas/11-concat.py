#!/usr/bin/env python3
'''
This module provides `concat` function
'''
import pandas as pd


def concat(df1, df2):
    '''
    takes two pd.DataFrame objects and:
    Indexes both dataframes on their Timestamp columns.
    Includes all timestamps from df2 (bitstamp)
    up to and including timestamp 1417411920.
    Concatenates the selected rows from df2 to the top of df1 (coinbase).
    Adds keys to the concatenated data,
    labeling the rows from df2 as bitstamp and the rows from df1 as coinbase.
    You should use index = __import__('10-index').index
    Returns the concatenated pd.DataFrame.
    '''
    index = __import__('10-index').index

    df1 = index(df1)
    df2 = index(df2)

    result = pd.concat(
        [df2.loc[df2.index <= 1417411920], df1],
        keys=["bitstamp", "coinbase"]
    )

    return result

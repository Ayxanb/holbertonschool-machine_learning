#!/usr/bin/env python3

'''
This module provides `hierarchy` function
'''

import pandas as pd


def hierarchy(df1, df2):
    '''
    takes two pd.DataFrame objects and:

    Rearranges the MultiIndex so that Timestamp is the first level.
    Concatenates the bitstamp and coinbase tables from timestamps 1417411980 to 1417417980, inclusive.
    Adds keys to the data, labeling rows from df2 as bitstamp and rows from df1 as coinbase.
    Ensures the data is displayed in chronological order.
    Returns: the concatenated pd.DataFrame.
    '''

    index = __import__('10-index').index

    df1 = index(df1)
    df2 = index(df2)

    df1 = df1.loc[(df1.index >= 1417411980) & (df1.index <= 1417417980)]
    df2 = df2.loc[(df2.index >= 1417411980) & (df2.index <= 1417417980)]

    result = pd.concat(
        [df2, df1],
        keys=["bitstamp", "coinbase"]
    )

    return result.swaplevel(0, 1).sort_index()

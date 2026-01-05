#!/usr/bin/env python3

''' this is a docstring :-) '''


def slice(df):
    '''
    takes a pd.DataFrame and:

    Extracts the columns High, Low, Close, and Volume_BTC.
    Selects every 60th row from these columns.
    Returns: the sliced pd.DataFrame
    '''

    cols = ["High", "Low", "Close", "Volume_(BTC)"]
    return df.loc[df.index % 60 == 0, cols]

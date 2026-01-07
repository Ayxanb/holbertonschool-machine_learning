#!/usr/bin/env python3

'''
This module provides `prune` function
'''


def prune(df):
    '''
    takes a pd.DataFrame and:

    Removes any entries where Close has NaN values.
    Returns: the modified pd.DataFrame.
    '''

    return df.dropna(subset=['Close'])

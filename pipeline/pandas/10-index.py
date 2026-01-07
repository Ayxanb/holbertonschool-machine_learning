#!/usr/bin/env python3

'''
This module provides `index` function
'''


def index(df):
    '''
    takes a pd.DataFrame and:

    Sets the Timestamp column as the index of the dataframe.
    Returns: the modified pd.DataFrame.
    '''

    return df.set_index("Timestamp")

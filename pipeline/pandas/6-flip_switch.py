#!/usr/bin/env python3

'''
This module contains `flip_switch` function
'''


def flip_switch(df):
    '''
    Sorts the DataFrame in reverse chronological order by its index,
    then transposes it.
    '''
    return df.sort_index(ascending=False).T

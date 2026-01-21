#!/usr/bin/env python3

'''
This module contains `add_arrays` function
'''


def add_arrays(arr1, arr2):
    '''
    adds two arrays, if shapes don't match, returns None
    '''

    if len(arr1) != len(arr2):
        return None

    return [a + b for a, b in zip(arr1, arr2)]

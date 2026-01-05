#!/usr/bin/env python3
''' This module provides a function to load data from file '''


import pandas as pd


def from_file(filename, delimiter):
    ''' this function returns the loaded data from file '''
    return pd.read_csv(filename, sep=delimiter)

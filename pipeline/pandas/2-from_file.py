''' This module provides a function to load data from file '''


import pandas as pd


def from_file(filename, delimiter):
    return pd.read_csv(filename, sep=delimiter)

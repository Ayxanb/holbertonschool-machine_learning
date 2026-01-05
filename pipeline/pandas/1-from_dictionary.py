#!/usr/bin/env python3
''' This module creates a pd.DataFrame object from a Python dictionary  '''


import pandas as pd


df = pd.DataFrame(
    {
        "First": pd.Series([0.0, 0.5, 1.0, 1.5])
        "Second": pd.Series(['one', 'two', 'three', 'four'])
    },
    columns=list('ABCD')
)

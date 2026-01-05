#!/usr/bin/env python3
''' This module creates a pd.DataFrame object from a Python dictionary  '''


import pandas as pd


df = pd.DataFrame(
    {
        "First": pd.Series([1], index=list('AB')),
        "Second": pd.Series([2])
    },
)

print(df)

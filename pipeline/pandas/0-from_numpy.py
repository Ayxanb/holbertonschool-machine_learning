#!/usr/bin/env python3

""" I hate docstring... """

import numpy as np, pandas as pd


def from_numpy(array):
    """ I hate docstring... """

    result = pd.DataFrame(array, columns=map(lambda x:chr(x), range(65, 91)))


    return result

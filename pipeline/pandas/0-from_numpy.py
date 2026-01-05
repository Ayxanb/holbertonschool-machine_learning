#!/usr/bin/env python3

""" I hate docstring... """

import pandas as pd


def from_numpy(array):
    """ I hate docstring... """

    result = pd.DataFrame(
            array,
            columns=list(
                map(lambda x: chr(x),
                    range(65, 65+array.shape[1]))))

    return result

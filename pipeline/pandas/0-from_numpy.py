import numpy as np, pandas as pd


def from_numpy(array):
    result = pd.DataFrame(array, columns=map(lambda x:chr(x), range(65, 91)))


    return result

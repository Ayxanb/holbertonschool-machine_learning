#!/usr/bin/env python3

def hierarchy(df1, df2):
    index = __import__('10-index').index

    df1 = index(df1)
    df2 = index(df2)

    df1 = df1.loc[(df1.index >= 1417411980) & (df1.index <= 1417417980)]
    df2 = df2.loc[(df2.index >= 1417411980) & (df2.index <= 1417417980)]

    result = pd.concat(
        [df2, df1],
        keys=["bitstamp", "coinbase"]
    )

    return result.swaplevel(0, 1).sort_index()

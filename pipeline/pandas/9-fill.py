#!/usr/bin/env python3

'''
This module provides `fill` function
'''


def fill(df):
    '''
    takes a pd.DataFrame and:

    Removes the Weighted_Price column.
    Fills missing values in the Close column with the previous rows value.
    Fills missing values in the High, Low, and Open columns with
    the corresponding Close value in the same row.
    Sets missing values in Volume_(BTC) and Volume_(Currency) to 0.
    Returns: the modified pd.DataFrame.
    '''

    df = df.copy()

    if "Weighted_Price" in df.columns:
        df = df.drop(columns=["Weighted_Price"])

    df['Close'] = df['Close'].ffill()

    for col in ["Open", "High", "Low"]:
        df[col] = df[col].fillna(df["Close"])

    for col in ["Volume_(BTC)", "Volume_(Currency)"]:
        df[col] = df[col].fillna(0)

    return df

#!/usr/bin/env python3

'''
This module provides `visualize` function
'''

import pandas as pd


def visualize():
    ''' i dont know what to put here :/ '''

    df = df.drop(columns=["Weighted_Price"])
    df = df.rename(columns={"Timestamp": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.set_index("Date")
    df["Close"] = df["Close"].ffill()

    for col in ["Open", "High", "Low"]:
        df[col] = df[col].fillna(df["Close"])

    for col in ["Volume_(BTC)", "Volume_(Currency)"]:
        df[col] = df[col].fillna(0)

    df = df.loc["2017-01-01":]


    df_daily = df.resample("D").agg({
        "High": "max",
        "Low": "min",
        "Open": "mean",
        "Close": "mean",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum"
    })

    return df_daily

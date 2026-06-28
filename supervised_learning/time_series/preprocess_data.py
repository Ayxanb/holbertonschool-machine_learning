# preprocess_data.py
import numpy as np
import pandas as pd


def preprocess_crypto_data(file_path, output_path):
    """Loads, cleans, resamples, and normalizes BTC historical data."""
    # Load dataset
    df = pd.read_csv(file_path)

    # Convert Unix timestamp to datetime and set as index
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df.set_index("Timestamp", inplace=True)

    # Handle missing values by forward filling prices
    df["Close"] = df["Close"].ffill()
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)

    # Resample to 1-hour intervals
    # Close price at the end of the hour, total volume during the hour
    df_hourly = df.resample("1h").agg(
        {"Close": "last", "Volume_(BTC)": "sum"}
    )

    # Drop any remaining NaNs at the very beginning
    df_hourly.dropna(inplace=True)

    # Min-Max Normalization (Scale features between 0 and 1)
    close_min, close_max = (
        df_hourly["Close"].min(),
        df_hourly["Close"].max(),
    )
    vol_min, vol_max = (
        df_hourly["Volume_(BTC)"].min(),
        df_hourly["Volume_(BTC)"].max(),
    )

    df_hourly["Close"] = (df_hourly["Close"] - close_min) / (
        close_max - close_min
    )
    df_hourly["Volume_(BTC)"] = (df_hourly["Volume_(BTC)"] - vol_min) / (
        vol_max - vol_min
    )

    # Save preprocessed data along with scaling metadata
    np.savez(
        output_path,
        data=df_hourly.values,
        meta=np.array([close_min, close_max, vol_min, vol_max]),
    )
    print(f"Preprocessed data saved successfully to {output_path}")


if __name__ == "__main__":
    # Adjust paths as needed for coinbase/bitstamp CSVs
    preprocess_crypto_data(
        "coinbaseUSD_1-min_data_2012-01-01_to_2021-03-31.csv",
        "preprocessed_btc.npz"
    )

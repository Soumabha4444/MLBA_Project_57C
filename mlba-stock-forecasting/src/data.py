import argparse
from datetime import datetime
import pandas as pd
import yfinance as yf
import os

def download_price(ticker, start, end, out_csv):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.reset_index().rename(columns={'Date':'date'})
    df['ticker'] = ticker
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved prices to {out_csv}")
    return df

if __name__ == "__main__":
    download_price("AAPL", "2008-01-01", "2016-10-31", "data/raw/aapl.csv")

# This downloads real AAPL stock data and saves it as data/raw/aapl.csv

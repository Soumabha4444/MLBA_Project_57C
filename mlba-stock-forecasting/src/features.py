import argparse
import os
import pandas as pd
import numpy as np

def create_features(df, lags=(1,2,3,5,10,20), roll_mean_win=5, roll_std_win=20):
    # normalize date column
    if 'Date' in df.columns and 'date' not in df.columns:
        df = df.rename(columns={'Date':'date'})
    if 'date' not in df.columns:
        raise ValueError("Input dataframe must contain a 'date' or 'Date' column.")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # standardize some common column names (map spaces -> underscores)
    to_check = ['Open','High','Low','Close','Adj Close','Adj_Close','Volume']
    for col in to_check:
        if col in df.columns:
            std_name = col.replace(' ', '_')
            df[std_name] = pd.to_numeric(df[col], errors='coerce')

    # prefer adjusted close if Close missing
    if 'Close' not in df.columns and 'Adj_Close' in df.columns:
        df['Close'] = df['Adj_Close']

    # accept lowercase variant
    if 'close' in df.columns and 'Close' not in df.columns:
        df['Close'] = pd.to_numeric(df['close'], errors='coerce')

    # sort and drop rows missing Close
    df = df.sort_values('date').copy().reset_index(drop=True)
    df = df.dropna(subset=['Close']).reset_index(drop=True)

    # create price-based target: next-day close (kept for reference)
    df['close_next'] = df['Close'].shift(-1)

    # next-day log return target (preferred for modeling)
    # ret_next = log(close_next / Close)
    df['ret_next'] = np.log(df['close_next'] / df['Close'])

    # log return (today)
    df['ret'] = np.log(df['Close']).diff()

    # lag features
    for l in lags:
        df[f'ret_lag_{l}'] = df['ret'].shift(l)

    # rolling stats
    df[f'ret_roll_{roll_mean_win}_mean'] = df['ret'].rolling(roll_mean_win).mean()
    df[f'ret_roll_{roll_std_win}_std'] = df['ret'].rolling(roll_std_win).std()

    # list of columns required for modelling
    cols_needed = ['ret_next'] + [f'ret_lag_{l}' for l in lags] + [f'ret_roll_{roll_mean_win}_mean', f'ret_roll_{roll_std_win}_std']

    df = df.dropna(subset=cols_needed).reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows remain after feature creation. Check input data length and column names.")

    return df

def save_sample(df, out_csv, n=200):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.head(n).to_csv(out_csv, index=False)
    print("Saved sample slice to", out_csv)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="infile", required=False, default="data/raw/aapl.csv",
                        help="Input raw CSV file (has Date/Close/Adj Close etc.)")
    parser.add_argument("--out", dest="outfile", required=False, default="data/processed/aapl_features.parquet")
    parser.add_argument("--sample_out", dest="sample_out", default="data/sample_slice.csv")
    parser.add_argument("--sample_n", type=int, default=200)
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        raise SystemExit(f"Input file not found: {args.infile}")
    df = pd.read_csv(args.infile)
    df_feats = create_features(df)
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    df_feats.to_parquet(args.outfile, index=False)
    save_sample(df_feats, args.sample_out, n=args.sample_n)
    print(f"Saved features to {args.outfile} with {len(df_feats)} rows")

if __name__ == "__main__":
    main()

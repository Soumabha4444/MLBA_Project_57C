#!/usr/bin/env python3
"""
build_text_embeddings.py

Robust SBERT embedding builder for a news CSV.

Features:
 - Detects date column (configurable with --date_col). Falls back to common names.
 - If date column is an integer (epoch), converts to datetime (configurable unit).
 - Allows specifying text column name (--text_col).
 - Computes SBERT embeddings for entire dataset in batches.
 - Saves parquet with columns: date (datetime normalized to date), text_emb (list of floats), optionally other meta.
"""

import argparse
from pathlib import Path
from typing import Optional, List
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm

# sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("ERROR: sentence_transformers not available. Install with `pip install sentence-transformers`.", file=sys.stderr)
    raise

COMMON_DATE_COLS = ["date", "created_utc", "timestamp", "created", "publish_date", "published_at", "time"]

def detect_date_col(df: pd.DataFrame, prefer: Optional[str] = None) -> Optional[str]:
    if prefer and prefer in df.columns:
        return prefer
    for c in COMMON_DATE_COLS:
        if c in df.columns:
            return c
    # otherwise try to find a datetime-like column
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    # try numeric columns that look like epoch (very large ints)
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
            series = df[c].dropna()
            if len(series) == 0:
                continue
            v = series.iloc[0]
            # heuristic: epoch seconds are > 1e9 (since 2001) and < 1e11
            if isinstance(v, (int, float)) and (1e9 < abs(v) < 1e12):
                return c
    return None

def parse_dates_column(series: pd.Series, unit: str = "s") -> pd.Series:
    # If already datetime, return normalized datetimes
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series).dt.normalize()
    # If numeric, treat as epoch
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        # use pandas to_datetime with unit
        return pd.to_datetime(series, unit=unit, errors="coerce").dt.normalize()
    # otherwise try parsing strings
    return pd.to_datetime(series, errors="coerce").dt.normalize()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--news_csv", required=True, help="Input CSV of news (must contain text column)")
    p.add_argument("--out", required=True, help="Output parquet path (will include columns 'date','text_emb')")
    p.add_argument("--text_col", default=None, help="Column name containing the text (title/body). If not set, tries common choices.")
    p.add_argument("--date_col", default=None, help="Column name containing date/timestamp. If not set, auto-detects.")
    p.add_argument("--epoch_unit", choices=["s", "ms"], default="s", help="If date column is numeric epoch, unit (seconds 's' or milliseconds 'ms').")
    p.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2", help="SBERT model name")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--text_cols_try", nargs="*", default=["title", "headline", "body", "content", "text"], help="Text columns to try if --text_col not provided")
    args = p.parse_args()

    path = Path(args.news_csv)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    # Read a small sample first to inspect columns
    sample = pd.read_csv(path, nrows=10)
    print("CSV columns:", sample.columns.tolist())

    # decide text column
    text_col = args.text_col
    if text_col is None:
        for c in args.text_cols_try:
            if c in sample.columns:
                text_col = c
                break
    if text_col is None:
        # fallback: choose first object dtype column that's not the detected date_col
        for c in sample.columns:
            if sample[c].dtype == object:
                text_col = c
                break
    if text_col is None:
        raise ValueError("Could not infer text column. Provide --text_col explicitly.")
    print("Using text column:", text_col)

    # read full CSV (we'll not parse dates yet)
    df = pd.read_csv(path)
    print("Read full CSV shape:", df.shape)

    # detect date column
    date_col = args.date_col or detect_date_col(df)
    if date_col is None:
        print("No date column detected; resulting parquet will have NaT for 'date'. You can pass --date_col to specify.")
        df["date_parsed"] = pd.NaT
    else:
        print("Using date column:", date_col)
        df["date_parsed"] = parse_dates_column(df[date_col], unit=args.epoch_unit)

    # normalize to midnight (date-only)
    df["date_parsed"] = pd.to_datetime(df["date_parsed"], errors="coerce").dt.normalize()

    # build a simple text series (concatenate title+body if both exist)
    # If text_col points to a combined field, use it
    text_series = df[text_col].astype(str).fillna("").values
    # If there's a 'body' column and text_col is only title, try to combine
    if "body" in df.columns and text_col != "body":
        # prefer title + body if available
        text_series = (df[text_col].fillna("").astype(str) + ". " + df["body"].fillna("").astype(str)).values

    # initialize SBERT model
    print("Loading SBERT model:", args.model_name)
    model = SentenceTransformer(args.model_name, device=args.device)

    # compute embeddings in batches
    n = len(text_series)
    batch_size = max(1, args.batch_size)
    embeddings: List[np.ndarray] = []
    print(f"Computing embeddings for {n} rows with batch_size={batch_size} on device={args.device}")
    for i in tqdm(range(0, n, batch_size)):
        batch_texts = text_series[i : i + batch_size].tolist()
        emb = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)  # shape (n, emb_dim)
    print("Embeddings shape:", embeddings.shape)

    # attach embeddings (as python lists) to dataframe
    df_out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date_parsed"]).dt.normalize(),
            "text_emb": [e.tolist() for e in embeddings],
        }
    )

    # if you want to keep other metadata (e.g., title, url), add them:
    # df_out["title"] = df.get("title", pd.Series([""]*len(df)))
    # df_out["url"] = df.get("url", pd.Series([""]*len(df)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    print("Wrote embeddings to", out_path)

if __name__ == "__main__":
    main()

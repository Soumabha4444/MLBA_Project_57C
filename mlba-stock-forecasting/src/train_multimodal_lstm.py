#!/usr/bin/env python3
"""
train_multimodal_lstm.py

Updated version with:
 - robust embedding alignment + aggregation
 - overlap statistics printing
 - emb_fill modes: zero, ffill, hybrid (ffill up to N then zero)
 - robust target detection / computation
 - fixed shape handling for sequences (no incorrect reshape)
 - debug prints
"""

import argparse
from pathlib import Path
import json
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# PyTorch (if not needed, you can change to any other framework)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -----------------------
# Helpers
# -----------------------
COMMON_TARGET_NAMES = ["target", "y", "ret", "returns", "target_return", "next_return"]
COMMON_DATE_COLS = ["date", "created_utc", "timestamp", "created", "publish_date", "published_at", "time"]


def detect_date_col(df: pd.DataFrame, prefer: Optional[str] = None) -> Optional[str]:
    if prefer and prefer in df.columns:
        return prefer
    for c in COMMON_DATE_COLS:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
            series = df[c].dropna()
            if len(series) == 0:
                continue
            v = series.iloc[0]
            if isinstance(v, (int, float)) and (1e9 < abs(v) < 1e12):
                return c
    return None


def parse_dates_column(series: pd.Series, unit: str = "s") -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series).dt.normalize()
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return pd.to_datetime(series, unit=unit, errors="coerce").dt.normalize()
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def align_embeddings_to_trade_dates(
    df_num: pd.DataFrame,
    df_text: pd.DataFrame,
    emb_agg: str = "mean",
    emb_fill: str = "zero",
    emb_ffill_limit: int = 0,
    debug: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """
    Align text embeddings to the dates present in df_num (trading dates).
    Returns (df_out, emb_dim) where df_out has columns ['date','text_emb'] (text_emb is list of floats).
    emb_fill: 'zero' | 'ffill' | 'hybrid' (hybrid == ffill up to emb_ffill_limit then zeros)
    emb_agg: aggregation over multiple text items per date: mean|sum|max
    """
    df_num = df_num.copy()
    df_num["date"] = pd.to_datetime(df_num["date"]).dt.normalize()
    df_text = df_text.copy()
    if "date" in df_text.columns:
        df_text["date"] = pd.to_datetime(df_text["date"]).dt.normalize()
    else:
        # ensure there's at least a date column (maybe named differently)
        date_col = detect_date_col(df_text)
        if date_col:
            df_text["date"] = pd.to_datetime(df_text[date_col]).dt.normalize()
        else:
            df_text["date"] = pd.NaT

    if "text_emb" not in df_text.columns:
        raise ValueError("text embeddings dataframe must contain 'text_emb' column")

    # Convert list/array to numpy arrays and drop invalids
    def to_arr(x):
        try:
            if isinstance(x, (list, tuple, np.ndarray)):
                arr = np.asarray(x, dtype=float)
                return arr
            # sometimes stored as string representation
            if isinstance(x, str):
                # try to eval safely: fallback - not ideal but often necessary
                import ast

                v = ast.literal_eval(x)
                arr = np.asarray(v, dtype=float)
                return arr
        except Exception:
            return None
        return None

    df_text["_emb_arr"] = df_text["text_emb"].apply(to_arr)
    df_text = df_text[~df_text["_emb_arr"].isna()].copy()
    if df_text.shape[0] == 0:
        raise ValueError("No valid embeddings found in text embeddings dataframe (column 'text_emb')")

    emb_dim = df_text["_emb_arr"].iloc[0].shape[0]
    # keep only rows with matching embedding dim
    df_text = df_text[df_text["_emb_arr"].apply(lambda x: x.shape[0] == emb_dim)].copy()

    # Build DataFrame with one column per embedding dimension indexed by date
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(df_text["_emb_arr"].tolist(), columns=emb_cols, index=df_text["date"])
    emb_df.index.name = "date"

    # aggregate multiple items per date
    if emb_agg == "mean":
        emb_daily = emb_df.groupby(emb_df.index).mean()
    elif emb_agg == "sum":
        emb_daily = emb_df.groupby(emb_df.index).sum()
    elif emb_agg == "max":
        emb_daily = emb_df.groupby(emb_df.index).max()
    else:
        emb_daily = emb_df.groupby(emb_df.index).mean()

    # alignment index (we align to stock feature dates)
    stock_dates = pd.to_datetime(df_num["date"].unique()).normalize()
    emb_index = pd.DatetimeIndex(sorted(stock_dates))
    emb_daily = emb_daily.reindex(emb_index)  # missing dates -> NaNs

    # Overlap stats
    unique_stock_dates = len(stock_dates)
    unique_emb_dates = emb_daily.dropna(how="all").shape[0]
    overlap = len(set(stock_dates).intersection(set(emb_daily.dropna(how="all").index)))
    if debug:
        print("=== Embedding/Feature date overlap statistics ===")
        print("Unique stock feature dates:", unique_stock_dates)
        print("Unique embedding dates:", unique_emb_dates)
        print(f"Overlap (stock ∩ emb): {overlap} ({overlap/unique_stock_dates*100:.2f}% of stock dates)")
        stock_dt_sorted = sorted(stock_dates)
        emb_dt_sorted = sorted(emb_daily.dropna(how="all").index)
        print("Example overlap dates (first 10):", list(sorted(set(stock_dt_sorted).intersection(set(emb_dt_sorted))))[:10])
        print("Example stock-only dates (first 10):", stock_dt_sorted[:10])
        print("Example emb-only dates (first 10):", emb_dt_sorted[:10])
        print("emb_fill:", emb_fill, "emb_ffill_limit:", emb_ffill_limit)
        print("===============================================")

    # Apply filling strategy
    if emb_fill == "zero":
        emb_filled = emb_daily.fillna(0.0)
    elif emb_fill == "ffill":
        emb_filled = emb_daily.ffill(limit=emb_ffill_limit)
    elif emb_fill in ("hybrid", "ffill_then_zero"):
        emb_filled = emb_daily.ffill(limit=emb_ffill_limit).fillna(0.0)
    else:
        emb_filled = emb_daily.fillna(0.0)

    # Convert back to list column in the same order as df_num's dates
    df_dates_order = pd.DataFrame({"date": pd.to_datetime(df_num["date"]).dt.normalize()})
    # Use emb_filled rows by df_dates_order order (for index alignment)
    # If some dates are outside emb index, reindex will produce NaNs -> fill as per above logic
    emb_for_dates = emb_filled.reindex(df_dates_order["date"].values).fillna(0.0)
    emb_list = emb_for_dates.values.tolist()
    df_out = df_dates_order.copy()
    df_out["text_emb"] = emb_list
    return df_out, emb_dim


def detect_or_compute_target(df: pd.DataFrame, target_col_arg: Optional[str], debug: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    Ensure df has a 'target' column. If none present and price column exists, compute next-day pct change.
    Returns (df, target_col_name)
    """
    df = df.copy()
    if target_col_arg and target_col_arg in df.columns:
        if debug:
            print(f"Using provided target column: {target_col_arg}")
        return df, target_col_arg

    for c in COMMON_TARGET_NAMES:
        if c in df.columns:
            if debug:
                print(f"Found existing target column: {c}")
            return df, c

    # try to compute from price columns
    price_col = None
    if "adj_close" in df.columns:
        price_col = "adj_close"
    elif "close" in df.columns:
        price_col = "close"
    elif "Close" in df.columns:
        price_col = "Close"

    if price_col:
        if debug:
            print(f"Computing target as next-day pct change of price col '{price_col}'")
        df = df.sort_values("date").copy()
        df["target"] = df[price_col].pct_change().shift(-1)  # next-day return
        return df, "target"

    raise ValueError("No 'target' column found in features parquet. Provide --target_col or include a price column like 'close' or 'adj_close' to compute one.")


# -----------------------
# Simple LSTM model
# -----------------------
class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        # use last timestep
        last = out[:, -1, :]
        y = self.head(last)
        return y.squeeze(-1)


# -----------------------
# Sequence creation
# -----------------------
def create_sequences(features_df: pd.DataFrame, seq_len: int, feature_cols: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build sequences from a dataframe sorted by date.
    Returns (X, y, df_sorted) where:
     - X shape = (n_samples, seq_len, n_features)
     - y shape = (n_samples,)
    """
    df = features_df.sort_values("date").reset_index(drop=True).copy()
    vals = df[feature_cols].values.astype(float)
    targets = df[target_col].values.astype(float)

    n = len(df)
    seqs = []
    ys = []
    indices = []
    for i in range(n - seq_len):
        seq = vals[i : i + seq_len]
        tgt = targets[i + seq_len]  # predict next day's target
        if np.isnan(tgt):
            continue
        seqs.append(seq)
        ys.append(tgt)
        indices.append(i)
    if len(seqs) == 0:
        return np.zeros((0, seq_len, len(feature_cols))), np.zeros((0,)), df
    X = np.stack(seqs, axis=0)
    y = np.array(ys, dtype=float)
    return X, y, df


# -----------------------
# Main
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="Features parquet with 'date' and feature columns")
    p.add_argument("--text_emb", required=True, help="Text embeddings parquet with 'date' and 'text_emb' (list) columns")
    p.add_argument("--seq_len", type=int, default=10)
    p.add_argument("--emb_agg", default="mean", choices=["mean", "sum", "max"])
    p.add_argument("--emb_fill", default="zero", choices=["zero", "ffill", "hybrid"])
    p.add_argument("--emb_ffill_limit", type=int, default=0, help="Days to forward-fill embeddings for ffill/hybrid")
    p.add_argument("--target_col", default=None, help="Name of target column if present")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--out", default="outputs/multimodal_lstm_preds.json")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--flatten_for_classical", action="store_true", help="If set, flatten sequences into single vectors (seq_len * features) for classical ML models. If not set, keep (batch,seq_len,features) for LSTM.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if args.debug:
        print("Using device:", device)

    # Load features & embeddings
    features_path = Path(args.features)
    text_emb_path = Path(args.text_emb)
    if not features_path.exists():
        raise FileNotFoundError(args.features)
    if not text_emb_path.exists():
        raise FileNotFoundError(args.text_emb)

    if args.debug:
        print("Loading features from:", features_path)
    df_features = pd.read_parquet(features_path)
    if args.debug:
        print("Loading embeddings from:", text_emb_path)
    df_text = pd.read_parquet(text_emb_path)

    # Ensure date exists in features
    if "date" not in df_features.columns:
        date_col = detect_date_col(df_features)
        if date_col:
            df_features["date"] = pd.to_datetime(df_features[date_col]).dt.normalize()
        else:
            raise ValueError("Features parquet must contain a 'date' column or a detectable datetime column.")

    # Ensure text date exists
    if "date" not in df_text.columns:
        date_col = detect_date_col(df_text)
        if date_col:
            df_text["date"] = pd.to_datetime(df_text[date_col]).dt.normalize()
        else:
            df_text["date"] = pd.NaT

    # Align embeddings -> df_aligned has text_emb for every stock date in df_features order
    if args.debug:
        print("Aligning/aggregating embeddings to trading dates...")
    df_text_aligned, emb_dim = align_embeddings_to_trade_dates(
        df_num=df_features,
        df_text=df_text,
        emb_agg=args.emb_agg,
        emb_fill=args.emb_fill,
        emb_ffill_limit=args.emb_ffill_limit,
        debug=args.debug,
    )
    if args.debug:
        print(f"Detected embedding dimension: {emb_dim}")

    # Merge aligned embeddings into df_features (keep df_features order)
    df_features = df_features.sort_values("date").reset_index(drop=True)
    df_text_aligned = df_text_aligned.reset_index(drop=True)
    if len(df_text_aligned) != len(df_features):
        # safer align by date merge on 'date' preserving df_features order
        df_merged = pd.merge(df_features, df_text_aligned, on="date", how="left", sort=False)
    else:
        df_merged = pd.concat([df_features.reset_index(drop=True), df_text_aligned["text_emb"].reset_index(drop=True)], axis=1)

    # If some text_emb are still NaN (shouldn't after filling), convert to zeros
    if df_merged["text_emb"].isna().any():
        if args.debug:
            print("Warning: some text_emb are NaN after alignment; filling with zeros.")
        df_merged["text_emb"] = df_merged["text_emb"].apply(lambda x: [0.0] * emb_dim if (pd.isna(x) or x is None) else x)

    # Detect or compute target
    df_merged, target_col = detect_or_compute_target(df_merged, args.target_col, debug=args.debug)

    # Build feature columns: numeric columns in features + embedding dimensions expanded as separate columns
    non_feature_cols = set(["date", target_col, "text_emb"])
    candidate_feature_cols = [c for c in df_merged.columns if c not in non_feature_cols and np.issubdtype(df_merged[c].dtype, np.number)]
    # Expand embedding dims into separate columns emb_0, emb_1, ...
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    emb_vals = np.vstack(df_merged["text_emb"].apply(lambda x: np.asarray(x, dtype=float)).values)
    emb_df = pd.DataFrame(emb_vals, columns=emb_cols, index=df_merged.index)
    df_features_expanded = pd.concat([df_merged.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)

    feature_cols = candidate_feature_cols + emb_cols
    if args.debug:
        print("Feature columns used (sample):", feature_cols[:10], "... total:", len(feature_cols))

    # Create sequences
    X, y, df_sorted = create_sequences(df_features_expanded[["date"] + feature_cols + [target_col]], seq_len=args.seq_len, feature_cols=feature_cols, target_col=target_col)
    if args.debug:
        print("Created sequences:")
        print("X shape:", X.shape)
        print("y shape:", y.shape)

    if X.shape[0] == 0:
        raise ValueError("No sequences created (maybe too short dataset relative to seq_len or all NaN targets).")

    # Split train/test simple chronological split (80/20)
    n_samples = X.shape[0]
    train_n = int(n_samples * 0.8)
    X_train = X[:train_n]
    y_train = y[:train_n]
    X_test = X[train_n:]
    y_test = y[train_n:]

    # Optionally flatten for classical models (user requested earlier)
    if args.flatten_for_classical:
        # each sample -> vector of length seq_len * n_features
        n_samples_train, seq_len_local, n_feats_ts = X_train.shape
        X_train_flat = X_train.reshape(n_samples_train, seq_len_local * n_feats_ts)
        X_test_flat = X_test.reshape(X_test.shape[0], seq_len_local * n_feats_ts)
        # create data loaders from flat arrays
        train_dataset = TensorDataset(torch.from_numpy(X_train_flat).float(), torch.from_numpy(y_train).float())
        test_dataset = TensorDataset(torch.from_numpy(X_test_flat).float(), torch.from_numpy(y_test).float())
        # For simplicity we won't train an LSTM if flatten_for_classical is True;
        # user should plug in classical model. We'll run a trivial baseline (ridge-like) here if desired.
        if args.debug:
            print("flatten_for_classical requested: returning flattened arrays for classical model training.")
        # Save predictions placeholder: just predict mean of train targets on test
        preds = [float(np.mean(y_train))] * len(y_test)
        out_dict = {"preds": preds, "y_test": y_test.tolist(), "method": "baseline_mean_due_to_flatten_flag"}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out_dict, f)
        print("Wrote fallback predictions to", args.out)
        return

    # Training with PyTorch LSTM
    input_size = X_train.shape[2]
    model = SimpleLSTM(input_size=input_size, hidden_size=64, num_layers=1, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.debug:
        print("Starting training: epochs:", args.epochs, "train_samples:", len(train_ds), "test_samples:", len(test_ds))

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_ds)
        if args.debug:
            print(f"[Epoch {epoch+1}/{args.epochs}] train loss: {avg_loss:.6f}")

    # Evaluation: predict on test set
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            preds.extend(out.cpu().numpy().tolist())
            trues.extend(yb.numpy().tolist())

    preds = np.array(preds)
    trues = np.array(trues)

    # helper to compute metrics for a pair of arrays
    def compute_metrics(a_true: np.ndarray, a_pred: np.ndarray):
        mae_v = np.mean(np.abs(a_pred - a_true))
        rmse_v = np.sqrt(np.mean((a_pred - a_true) ** 2))
        if len(a_true) >= 2:
            da_v = 100.0 * np.mean(np.sign(a_pred) == np.sign(a_true))
        else:
            da_v = 0.0
        return mae_v, rmse_v, da_v

    # point estimates
    mae, rmse, da = compute_metrics(trues, preds)

    # bootstrap confidence intervals (95%) for MAE, RMSE, DA
    n_test_pts = len(trues)
    n_boot = 1000
    seed = 42
    if n_test_pts >= 2:
        rng = np.random.default_rng(seed)
        mae_bs = np.empty(n_boot, dtype=float)
        rmse_bs = np.empty(n_boot, dtype=float)
        da_bs = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n_test_pts, size=n_test_pts)  # with-replacement indices
            t_sample = trues[idx]
            p_sample = preds[idx]
            m, r, d = compute_metrics(t_sample, p_sample)
            mae_bs[i] = m
            rmse_bs[i] = r
            da_bs[i] = d
        # 2.5th and 97.5th percentiles
        mae_lo, mae_hi = np.percentile(mae_bs, [2.5, 97.5]).tolist()
        rmse_lo, rmse_hi = np.percentile(rmse_bs, [2.5, 97.5]).tolist()
        da_lo, da_hi = np.percentile(da_bs, [2.5, 97.5]).tolist()
    else:
        # not enough points to bootstrap reliably
        mae_lo = mae_hi = float(mae)
        rmse_lo = rmse_hi = float(rmse)
        da_lo = da_hi = float(da)

    out_metrics = {
        "MAE_mean": float(mae),
        "MAE_lo": float(mae_lo),
        "MAE_hi": float(mae_hi),
        "RMSE_mean": float(rmse),
        "RMSE_lo": float(rmse_lo),
        "RMSE_hi": float(rmse_hi),
        "DA_mean": float(da),
        "DA_lo": float(da_lo),
        "DA_hi": float(da_hi),
        "n_test": int(n_test_pts),
        "bootstrap_samples": int(n_boot) if n_test_pts >= 2 else 0,
        "bootstrap_seed": int(seed) if n_test_pts >= 2 else None,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"preds": preds.tolist(), "y_test": trues.tolist(), "metrics": out_metrics}, f)

    print("Saved predictions+metrics to", args.out)
    if args.debug:
        print("Metrics:", json.dumps(out_metrics, indent=2))


if __name__ == "__main__":
    main()

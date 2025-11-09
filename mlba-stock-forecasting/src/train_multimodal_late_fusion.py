#!/usr/bin/env python3
"""
train_multimodal_late_fusion.py

Late-fusion multimodal LSTM:
 - numeric features -> numeric LSTM encoder -> numeric MLP head
 - text embeddings  -> text LSTM encoder    -> text   MLP head
 - fusion: concat(numeric_repr, text_repr) -> fusion MLP -> scalar prediction

Usage mirrors your previous script (features parquet, text_emb parquet, seq_len, emb_agg, emb_fill, etc).
"""
import argparse
from pathlib import Path
import json
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -----------------------
# (Reuse helpers from your script)
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

def align_embeddings_to_trade_dates(
    df_num: pd.DataFrame,
    df_text: pd.DataFrame,
    emb_agg: str = "mean",
    emb_fill: str = "zero",
    emb_ffill_limit: int = 0,
    debug: bool = False,
) -> Tuple[pd.DataFrame, int]:
    # (copy of your align_embeddings_to_trade_dates, adapted minimally)
    df_num = df_num.copy()
    df_num["date"] = pd.to_datetime(df_num["date"]).dt.normalize()
    df_text = df_text.copy()
    if "date" in df_text.columns:
        df_text["date"] = pd.to_datetime(df_text["date"]).dt.normalize()
    else:
        date_col = detect_date_col(df_text)
        if date_col:
            df_text["date"] = pd.to_datetime(df_text[date_col]).dt.normalize()
        else:
            df_text["date"] = pd.NaT

    if "text_emb" not in df_text.columns:
        raise ValueError("text embeddings dataframe must contain 'text_emb' column")

    def to_arr(x):
        try:
            if isinstance(x, (list, tuple, np.ndarray)):
                return np.asarray(x, dtype=float)
            if isinstance(x, str):
                import ast
                v = ast.literal_eval(x)
                return np.asarray(v, dtype=float)
        except Exception:
            return None
        return None

    df_text["_emb_arr"] = df_text["text_emb"].apply(to_arr)
    df_text = df_text[~df_text["_emb_arr"].isna()].copy()
    if df_text.shape[0] == 0:
        raise ValueError("No valid embeddings found in text embeddings dataframe (column 'text_emb')")

    emb_dim = df_text["_emb_arr"].iloc[0].shape[0]
    df_text = df_text[df_text["_emb_arr"].apply(lambda x: x.shape[0] == emb_dim)].copy()

    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(df_text["_emb_arr"].tolist(), columns=emb_cols, index=df_text["date"])
    emb_df.index.name = "date"

    if emb_agg == "mean":
        emb_daily = emb_df.groupby(emb_df.index).mean()
    elif emb_agg == "sum":
        emb_daily = emb_df.groupby(emb_df.index).sum()
    elif emb_agg == "max":
        emb_daily = emb_df.groupby(emb_df.index).max()
    else:
        emb_daily = emb_df.groupby(emb_df.index).mean()

    stock_dates = pd.to_datetime(df_num["date"].unique()).normalize()
    emb_index = pd.DatetimeIndex(sorted(stock_dates))
    emb_daily = emb_daily.reindex(emb_index)

    if emb_fill == "zero":
        emb_filled = emb_daily.fillna(0.0)
    elif emb_fill == "ffill":
        emb_filled = emb_daily.ffill(limit=emb_ffill_limit)
    elif emb_fill in ("hybrid", "ffill_then_zero"):
        emb_filled = emb_daily.ffill(limit=emb_ffill_limit).fillna(0.0)
    else:
        emb_filled = emb_daily.fillna(0.0)

    df_dates_order = pd.DataFrame({"date": pd.to_datetime(df_num["date"]).dt.normalize()})
    emb_for_dates = emb_filled.reindex(df_dates_order["date"].values).fillna(0.0)
    emb_list = emb_for_dates.values.tolist()
    df_out = df_dates_order.copy()
    df_out["text_emb"] = emb_list
    return df_out, emb_dim

def detect_or_compute_target(df: pd.DataFrame, target_col_arg: Optional[str], debug: bool = False) -> Tuple[pd.DataFrame, str]:
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
        df["target"] = df[price_col].pct_change().shift(-1)
        return df, "target"

    raise ValueError("No 'target' column found in features parquet. Provide --target_col or include a price column like 'close' or 'adj_close' to compute one.")

# -----------------------
# Sequence creation for two modalities
# -----------------------
def create_sequences_two_modal(
    df: pd.DataFrame,
    seq_len: int,
    numeric_cols: List[str],
    emb_col: str,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Return (X_num, X_emb, y, df_sorted)
     - X_num: (n_samples, seq_len, n_numeric_feats)
     - X_emb: (n_samples, seq_len, emb_dim)
     - y:     (n_samples,)
    """
    df2 = df.sort_values("date").reset_index(drop=True).copy()
    num_vals = df2[numeric_cols].values.astype(float)
    emb_vals = np.vstack(df2[emb_col].apply(lambda x: np.asarray(x, dtype=float)).values)
    targets = df2[target_col].values.astype(float)

    n = len(df2)
    seqs_num = []
    seqs_emb = []
    ys = []
    for i in range(n - seq_len):
        seq_n = num_vals[i : i + seq_len]
        seq_e = emb_vals[i : i + seq_len]
        tgt = targets[i + seq_len]
        if np.isnan(tgt):
            continue
        seqs_num.append(seq_n)
        seqs_emb.append(seq_e)
        ys.append(tgt)
    if len(seqs_num) == 0:
        return np.zeros((0, seq_len, num_vals.shape[1])), np.zeros((0, seq_len, emb_vals.shape[1])), np.zeros((0,)), df2
    X_num = np.stack(seqs_num, axis=0)
    X_emb = np.stack(seqs_emb, axis=0)
    y = np.array(ys, dtype=float)
    return X_num, X_emb, y, df2

# -----------------------
# Model: Late fusion
# -----------------------
class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.proj = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU())

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)
        last = out[:, -1, :]  # (batch, hidden_size)
        return self.proj(last)  # (batch, hidden_size//2)

class LateFusionModel(nn.Module):
    def __init__(self, num_input_size: int, emb_input_size: int, enc_hidden: int = 64, head_hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.num_enc = LSTMEncoder(input_size=num_input_size, hidden_size=enc_hidden, num_layers=1, dropout=dropout)
        self.emb_enc = LSTMEncoder(input_size=emb_input_size, hidden_size=enc_hidden, num_layers=1, dropout=dropout)
        fusion_in = (enc_hidden//2) * 2
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x_num, x_emb):
        # x_num: (batch, seq_len, n_num_feats)
        # x_emb: (batch, seq_len, emb_dim)
        r_num = self.num_enc(x_num)
        r_emb = self.emb_enc(x_emb)
        fusion = torch.cat([r_num, r_emb], dim=1)
        out = self.fusion_head(fusion)
        return out.squeeze(-1)

# -----------------------
# Training & main
# -----------------------
def compute_metrics(a_true: np.ndarray, a_pred: np.ndarray):
    mae_v = np.mean(np.abs(a_pred - a_true))
    rmse_v = np.sqrt(np.mean((a_pred - a_true) ** 2))
    if len(a_true) >= 2:
        da_v = 100.0 * np.mean(np.sign(a_pred) == np.sign(a_true))
    else:
        da_v = 0.0
    return mae_v, rmse_v, da_v

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--text_emb", required=True)
    p.add_argument("--seq_len", type=int, default=10)
    p.add_argument("--emb_agg", default="mean", choices=["mean", "sum", "max"])
    p.add_argument("--emb_fill", default="zero", choices=["zero", "ffill", "hybrid"])
    p.add_argument("--emb_ffill_limit", type=int, default=0)
    p.add_argument("--target_col", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="outputs/multimodal_late_preds.json")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if args.debug:
        print("Device:", device)

    df_features = pd.read_parquet(args.features)
    df_text = pd.read_parquet(args.text_emb)

    if "date" not in df_features.columns:
        date_col = detect_date_col(df_features)
        if date_col:
            df_features["date"] = pd.to_datetime(df_features[date_col]).dt.normalize()
        else:
            raise ValueError("Features parquet must contain a 'date' column or a detectable datetime column.")

    if "date" not in df_text.columns:
        date_col = detect_date_col(df_text)
        if date_col:
            df_text["date"] = pd.to_datetime(df_text[date_col]).dt.normalize()
        else:
            df_text["date"] = pd.NaT

    df_text_aligned, emb_dim = align_embeddings_to_trade_dates(
        df_num=df_features,
        df_text=df_text,
        emb_agg=args.emb_agg,
        emb_fill=args.emb_fill,
        emb_ffill_limit=args.emb_ffill_limit,
        debug=args.debug,
    )

    # Merge aligned embeddings into features (preserve features order)
    df_features = df_features.sort_values("date").reset_index(drop=True)
    df_text_aligned = df_text_aligned.reset_index(drop=True)
    if len(df_text_aligned) != len(df_features):
        df_merged = pd.merge(df_features, df_text_aligned, on="date", how="left", sort=False)
    else:
        df_merged = pd.concat([df_features.reset_index(drop=True), df_text_aligned["text_emb"].reset_index(drop=True)], axis=1)

    if df_merged["text_emb"].isna().any():
        if args.debug:
            print("Filling NaN text_emb with zeros")
        df_merged["text_emb"] = df_merged["text_emb"].apply(lambda x: [0.0]*emb_dim if (pd.isna(x) or x is None) else x)

    df_merged, target_col = detect_or_compute_target(df_merged, args.target_col, debug=args.debug)

    non_feature_cols = set(["date", target_col, "text_emb"])
    numeric_cols = [c for c in df_merged.columns if c not in non_feature_cols and np.issubdtype(df_merged[c].dtype, np.number)]

    if args.debug:
        print("Numeric feature columns (sample):", numeric_cols[:10], " total:", len(numeric_cols))
        print("Embedding dim:", emb_dim)

    # create sequences for both modalities
    X_num, X_emb, y, df_sorted = create_sequences_two_modal(df_merged[["date"]+numeric_cols+["text_emb", target_col]],
                                                            seq_len=args.seq_len, numeric_cols=numeric_cols, emb_col="text_emb", target_col=target_col)
    if args.debug:
        print("X_num shape:", X_num.shape, "X_emb shape:", X_emb.shape, "y shape:", y.shape)

    if X_num.shape[0] == 0:
        raise ValueError("No sequences created (dataset too short or NaN targets)")

    # split chronologically
    n = X_num.shape[0]
    train_n = int(n*0.8)
    Xn_tr = X_num[:train_n]
    Xe_tr = X_emb[:train_n]
    y_tr = y[:train_n]

    Xn_te = X_num[train_n:]
    Xe_te = X_emb[train_n:]
    y_te = y[train_n:]

    # build model
    num_input_size = Xn_tr.shape[2]
    emb_input_size = X_emb.shape[2]
    model = LateFusionModel(num_input_size=num_input_size, emb_input_size=emb_input_size, enc_hidden=64, head_hidden=64, dropout=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(Xn_tr).float(), torch.from_numpy(Xe_tr).float(), torch.from_numpy(y_tr).float())
    test_ds = TensorDataset(torch.from_numpy(Xn_te).float(), torch.from_numpy(Xe_te).float(), torch.from_numpy(y_te).float())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.debug:
        print(f"Training for {args.epochs} epochs | train samples: {len(train_ds)} | test samples: {len(test_ds)}")

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for xb_num, xb_emb, yb in train_loader:
            xb_num = xb_num.to(device)
            xb_emb = xb_emb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb_num, xb_emb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb_num.size(0)
        avg_loss = total_loss / len(train_ds)
        if args.debug:
            print(f"[Epoch {epoch+1}/{args.epochs}] train loss: {avg_loss:.6f}")

    # eval
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb_num, xb_emb, yb in test_loader:
            xb_num = xb_num.to(device)
            xb_emb = xb_emb.to(device)
            out = model(xb_num, xb_emb)
            preds.extend(out.cpu().numpy().tolist())
            trues.extend(yb.numpy().tolist())

    preds = np.array(preds)
    trues = np.array(trues)

    mae, rmse, da = compute_metrics(trues, preds)

    # bootstrap CI (1000) as in your prior script
    n_test_pts = len(trues)
    n_boot = 1000
    seed = 42
    if n_test_pts >= 2:
        rng = np.random.default_rng(seed)
        mae_bs = np.empty(n_boot, dtype=float)
        rmse_bs = np.empty(n_boot, dtype=float)
        da_bs = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n_test_pts, size=n_test_pts)
            t_sample = trues[idx]
            p_sample = preds[idx]
            m, r, d = compute_metrics(t_sample, p_sample)
            mae_bs[i] = m
            rmse_bs[i] = r
            da_bs[i] = d
        mae_lo, mae_hi = np.percentile(mae_bs, [2.5, 97.5]).tolist()
        rmse_lo, rmse_hi = np.percentile(rmse_bs, [2.5, 97.5]).tolist()
        da_lo, da_hi = np.percentile(da_bs, [2.5, 97.5]).tolist()
    else:
        mae_lo = mae_hi = float(mae)
        rmse_lo = rmse_hi = float(rmse)
        da_lo = da_hi = float(da)

    out_metrics = {
        "MAE_mean": float(mae), "MAE_lo": float(mae_lo), "MAE_hi": float(mae_hi),
        "RMSE_mean": float(rmse), "RMSE_lo": float(rmse_lo), "RMSE_hi": float(rmse_hi),
        "DA_mean": float(da), "DA_lo": float(da_lo), "DA_hi": float(da_hi),
        "n_test": int(n_test_pts), "bootstrap_samples": int(n_boot) if n_test_pts>=2 else 0, "bootstrap_seed": int(seed) if n_test_pts>=2 else None
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"preds": preds.tolist(), "y_test": trues.tolist(), "metrics": out_metrics}, f)

    print("Saved predictions+metrics to", args.out)
    if args.debug:
        print("Metrics:", json.dumps(out_metrics, indent=2))

if __name__ == "__main__":
    main()

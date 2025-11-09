#!/usr/bin/env python3
# save as src/train_lstm.py
import argparse, os, json, math
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# PyTorch LSTM model
# ---------------------------
class SeqDataset(Dataset):
    def __init__(self, X, y):
        # X: (#samples, seq_len, n_features), y: (#samples,)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)           # out: (B, T, hidden)
        out = out[:, -1, :]             # take last timestep
        out = self.fc(out)              # (B,1)
        return out.squeeze(1)          # (B,)

# ---------------------------
# Utilities
# ---------------------------
def rmse(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def t_ci(x, alpha=0.95):
    x = np.array(x, dtype=float); n=len(x)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(x))
    if n == 1:
        return mean, mean, mean
    se = float(np.std(x, ddof=1)/math.sqrt(n))
    t = stats.t.ppf((1+alpha)/2., n-1)
    return mean, mean - t*se, mean + t*se

def directional_accuracy_percent(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return 0.0
    y_true = y_true[mask]; y_pred = y_pred[mask]
    return float(100.0 * np.mean(np.sign(y_true) == np.sign(y_pred)))

# ---------------------------
# Build sliding sequences
# ---------------------------
def build_sequences(df, features, seq_len):
    """
    Given sorted df, returns:
      indices: list of indices i for which sequence X[i] ends at row i (so target is at row i)
      X_seq: np.array (#samples, seq_len, n_features)
      y: np.array (#samples,) -> df.loc[i, 'ret_next']
    Only rows where full seq_len history exists are included.
    """
    n = len(df)
    X_list = []
    y_list = []
    idx_list = []
    arr = df[features].values
    targets = df['ret_next'].values
    for end_idx in range(seq_len - 1, n):
        start_idx = end_idx - (seq_len - 1)
        # ensure target exists and finite
        if not np.isfinite(targets[end_idx]):
            continue
        seq = arr[start_idx:end_idx+1]  # shape (seq_len, n_features)
        if np.any(~np.isfinite(seq)):
            continue
        X_list.append(seq)
        y_list.append(targets[end_idx])
        idx_list.append(end_idx)
    if len(X_list) == 0:
        return np.empty((0, seq_len, len(features))), np.empty((0,)), []
    return np.stack(X_list), np.array(y_list), idx_list

# ---------------------------
# Train per-fold LSTM
# ---------------------------
def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb = Xb.to(device); yb = yb.to(device)
        opt.zero_grad()
        out = model(Xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * Xb.size(0)
    return total_loss / len(loader.dataset)

def predict(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            out = model(Xb).cpu().numpy()
            preds.append(out)
            trues.append(yb.numpy())
    if len(preds) == 0:
        return np.array([]), np.array([])
    return np.concatenate(trues), np.concatenate(preds)

# ---------------------------
# Main: rolling folds
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/aapl_features.parquet")
    parser.add_argument("--out", default="outputs/lstm_preds.json")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    df = pd.read_parquet(args.features).sort_values('date').reset_index(drop=True)
    # choose features to feed LSTM per-timestep
    features = [c for c in df.columns if c.startswith('ret_lag_') or c.startswith('ret_roll_') or c in ['ret']]
    if len(features) == 0:
        raise SystemExit("No features found for LSTM. Check your processed parquet columns.")

    # rolling-origin folds (same as baseline)
    dates = sorted(df['date'].unique())
    initial_train_days=800; test_days=60; step_days=60
    folds=[]
    start_idx=0
    while True:
        train_end = start_idx + initial_train_days - 1
        test_start = train_end + 1
        test_end = test_start + test_days - 1
        if test_end >= len(dates): break
        folds.append((dates[0], dates[train_end], dates[test_start], dates[test_end]))
        start_idx += step_days

    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    out_recs = []

    for idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        print(f"[Fold {idx}] train_end={train_end}, test_start={test_start}, test_end={test_end}")
        # split df by date
        train_mask = df['date'] <= train_end
        test_mask = (df['date'] >= test_start) & (df['date'] <= test_end)
        df_train = df[train_mask].reset_index(drop=True)
        df_test = df[test_mask].reset_index(drop=True)

        # Build sequences from the full train portion to scale properly and then build for test using scaler
        X_train_seq, y_train_seq, idxs_train = build_sequences(df_train, features, args.seq_len)
        # for test sequences we need context that may include rows inside train - so build sequences on concatenated df up to test_end
        df_upto_test = df[df['date'] <= test_end].reset_index(drop=True)
        X_all_seq, y_all_seq, idxs_all = build_sequences(df_upto_test, features, args.seq_len)

        # identify which sequences correspond to test indices (those whose end idx is in df_upto_test and date between test_start/test_end)
        test_end_positions = []
        # map idxs_all (which are indices in df_upto_test) to dates; need to pick those with date in test range
        dates_all = df_upto_test['date'].values
        for pos, end_idx in enumerate(idxs_all):
            dt = dates_all[end_idx]
            if (dt >= np.datetime64(test_start)) and (dt <= np.datetime64(test_end)):
                test_end_positions.append(pos)

        # now prepare X_test_seq and y_test_seq
        if len(test_end_positions) == 0:
            print(f"  no test sequences for fold {idx}, skipping")
            continue
        X_test_seq = X_all_seq[test_end_positions]
        y_test_seq = y_all_seq[test_end_positions]

        # scale features: fit scaler on flattened training sequences (all timesteps)
        n_feats = len(features)
        if X_train_seq.shape[0] == 0:
            print("  no train sequences for this fold; skipping")
            continue
        # flatten to (num_samples * seq_len, n_feats)
        flat_train = X_train_seq.reshape(-1, n_feats)
        scaler = StandardScaler().fit(flat_train)
        # apply scaler
        X_train_seq_scaled = scaler.transform(flat_train).reshape(X_train_seq.shape)
        X_test_seq_scaled = scaler.transform(X_test_seq.reshape(-1, n_feats)).reshape(X_test_seq.shape)

        # build dataloaders
        train_ds = SeqDataset(X_train_seq_scaled, y_train_seq)
        test_ds = SeqDataset(X_test_seq_scaled, y_test_seq)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        # model
        model = LSTMRegressor(input_size=n_feats, hidden_size=args.hidden).to(device)
        loss_fn = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        # training loop
        for ep in range(args.epochs):
            tr_loss = train_epoch(model, train_loader, opt, loss_fn, device)
            if (ep+1) % 5 == 0 or ep == 0 or ep == args.epochs-1:
                print(f"   Epoch {ep+1}/{args.epochs}, train_loss={tr_loss:.6f}")

        # predict on test
        y_true, y_pred = predict(model, test_loader, device)
        # convert to plain floats
        y_true_list = [float(x) for x in y_true.tolist()]
        y_pred_list = [float(x) for x in y_pred.tolist()]

        mae_val = float(mean_absolute_error(y_true, y_pred))
        rmse_val = rmse(y_true, y_pred)
        da_val = directional_accuracy_percent(y_true, y_pred)

        # record
        out_rec = {
            "fold": int(idx),
            "model": "lstm_numerical",
            "train_start": pd.to_datetime(str(train_start)).isoformat(),
            "train_end": pd.to_datetime(str(train_end)).isoformat(),
            "test_start": pd.to_datetime(str(test_start)).isoformat(),
            "test_end": pd.to_datetime(str(test_end)).isoformat(),
            "y_true": y_true_list,
            "y_pred": y_pred_list,
            "metrics": {
                "MAE": mae_val,
                "RMSE": rmse_val,
                "DA_percent": da_val
            }
        }
        out_recs.append(out_rec)

    # summary across folds
    maes = [r["metrics"]["MAE"] for r in out_recs]
    rmses = [r["metrics"]["RMSE"] for r in out_recs]
    das = [r["metrics"]["DA_percent"] for r in out_recs]
    summary = {
        "MAE": {"mean": t_ci(maes)[0], "lo": t_ci(maes)[1], "hi": t_ci(maes)[2]},
        "RMSE": {"mean": t_ci(rmses)[0], "lo": t_ci(rmses)[1], "hi": t_ci(rmses)[2]},
        "DA_percent": {"mean": t_ci(das)[0], "lo": t_ci(das)[1], "hi": t_ci(das)[2]}
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_blob = {"per_fold": out_recs, "summary": summary}
    with open(args.out, "w") as f:
        json.dump(out_blob, f, indent=2)
    print("Saved LSTM preds+metrics to", args.out)

if __name__ == "__main__":
    main()

import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from scipy import stats

def rmse(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))

def directional_accuracy_returns(y_true, y_pred):
    """
    Directional accuracy for returns: sign(pred) == sign(true)
    Returns percentage in [0,100].
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return 0.0
    signs_pred = np.sign(y_pred[:n])
    signs_true = np.sign(y_true[:n])
    acc = np.mean(signs_pred == signs_true)
    return float(100.0 * acc)

def rolling_origin(df, initial_train_days=800, test_days=60, step_days=60):
    dates = sorted(df['date'].unique())
    folds=[]
    start_idx=0
    while True:
        train_end = start_idx + initial_train_days - 1
        test_start = train_end + 1
        test_end = test_start + test_days - 1
        if test_end >= len(dates): break
        folds.append((dates[0], dates[train_end], dates[test_start], dates[test_end]))
        start_idx += step_days
    return folds

def run_fold(df, train_end_date, test_start, test_end, features):
    train_df = df[df['date']<=train_end_date]
    test_df = df[(df['date']>=test_start) & (df['date']<=test_end)].copy().reset_index(drop=True)

    X_train = train_df[features].values
    y_train = train_df['ret_next'].values   # now predicting next-day log return
    X_test = test_df[features].values
    y_test = test_df['ret_next'].values

    # Keep current_close for reference (not used for DA on returns)
    current_close = test_df['Close'].values if 'Close' in test_df.columns else np.full_like(y_test, np.nan, dtype=float)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # compute metrics for this fold (on returns)
    mae_val = float(mean_absolute_error(y_test, preds))
    rmse_val = float(rmse(y_test, preds))
    da_val = directional_accuracy_returns(y_test, preds)

    # convert arrays to plain python lists (floats / None) for JSON
    def as_pylist(arr, cast=float):
        out = []
        for x in list(arr):
            if pd.isna(x) or (isinstance(x, float) and (np.isinf(x) or np.isnan(x))):
                out.append(None)
            else:
                out.append(cast(x))
        return out

    return {
        "y_true": as_pylist(y_test, float),
        "y_pred": as_pylist(preds, float),
        "current_close": as_pylist(current_close, float),
        "metrics": {
            "MAE": mae_val,
            "RMSE": rmse_val,
            "DA_percent": da_val
        }
    }

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=False, default="data/processed/aapl_features.parquet")
    parser.add_argument("--out", required=False, default="outputs/baseline_preds.json")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    df = df.sort_values('date').reset_index(drop=True)

    # pick feature columns (lags & rolling stats)
    features = [c for c in df.columns if c.startswith('ret_lag_') or c.startswith('ret_roll_')]

    folds = rolling_origin(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_preds=[]

    for idx,(train_start, train_end, test_start, test_end) in enumerate(folds):
        print(f"Fold {idx}: train_end={train_end}, test_start={test_start}, test_end={test_end}")
        rec = run_fold(df, train_end, test_start, test_end, features)
        # stringify timestamps
        def to_str(d):
            try:
                return d.isoformat()
            except Exception:
                return str(d)
        out_rec = {
            "fold": int(idx),
            "model": "ridge_baseline",
            "train_start": to_str(train_start),
            "train_end": to_str(train_end),
            "test_start": to_str(test_start),
            "test_end": to_str(test_end),
            "y_true": rec["y_true"],
            "y_pred": rec["y_pred"],
            "current_close": rec["current_close"],
            "metrics": rec["metrics"]
        }
        all_preds.append(out_rec)

    # compute summary across folds
    maes = [r["metrics"]["MAE"] for r in all_preds]
    rmses = [r["metrics"]["RMSE"] for r in all_preds]
    das = [r["metrics"]["DA_percent"] for r in all_preds]

    summary = {
        "MAE": { "mean": t_ci(maes)[0], "lo": t_ci(maes)[1], "hi": t_ci(maes)[2] },
        "RMSE": { "mean": t_ci(rmses)[0], "lo": t_ci(rmses)[1], "hi": t_ci(rmses)[2] },
        "DA_percent": { "mean": t_ci(das)[0], "lo": t_ci(das)[1], "hi": t_ci(das)[2] }
    }

    out_blob = {
        "per_fold": all_preds,
        "summary": summary
    }

    with open(args.out,"w") as f:
        json.dump(out_blob,f, indent=2)
    print("Saved preds and metrics to", args.out)

if __name__=="__main__":
    main()

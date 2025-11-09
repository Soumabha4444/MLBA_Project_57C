import json, argparse, os, math, sys
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

def t_confidence_interval(x, confidence=0.95):
    x = np.array(x, dtype=float)
    n = len(x)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(x))
    if n == 1:
        return mean, mean, mean
    se = float(np.std(x, ddof=1) / math.sqrt(n))
    t = stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, mean - t * se, mean + t * se

def directional_accuracy_filtered(y_true, y_pred):
    """
    Compute directional accuracy (%) for returns.
    Filters out non-finite entries and aligns arrays.
    """
    # convert to numpy arrays and attempt to coerce to float
    y_t = np.array([_safe_float(x) for x in y_true], dtype=float)
    y_p = np.array([_safe_float(x) for x in y_pred], dtype=float)

    # mask finite entries in both
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    if mask.sum() == 0:
        return 0.0
    y_t = y_t[mask]
    y_p = y_p[mask]
    signs_true = np.sign(y_t)
    signs_pred = np.sign(y_p)
    acc = np.mean(signs_true == signs_pred)
    return float(100.0 * acc)

def _safe_float(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float('nan')

def load_preds(path):
    blob = json.load(open(path))
    if isinstance(blob, dict) and "per_fold" in blob:
        records = blob["per_fold"]
    elif isinstance(blob, list):
        records = blob
    else:
        raise ValueError("Unrecognized preds_json structure. Expect list or dict with 'per_fold'.")
    return records

def t_confidence_interval_list(x):
    return t_confidence_interval(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = load_preds(args.preds_json)

    maes, rmses, das = [], [], []
    for rec in records:
        y_true = rec.get("y_true", [])
        y_pred = rec.get("y_pred", [])
        # compute per-fold metrics, making sure to coerce/filter invalid entries
        # MAE/RMSE: only use aligned finite pairs
        y_t = np.array([_safe_float(x) for x in y_true], dtype=float)
        y_p = np.array([_safe_float(x) for x in y_pred], dtype=float)
        mask = np.isfinite(y_t) & np.isfinite(y_p)
        if mask.sum() == 0:
            continue
        y_t_f = y_t[mask]
        y_p_f = y_p[mask]
        maes.append(float(mean_absolute_error(y_t_f, y_p_f)))
        rmses.append(float(math.sqrt(mean_squared_error(y_t_f, y_p_f))))
        das.append(directional_accuracy_filtered(y_t_f, y_p_f))

    mae_mean, mae_lo, mae_hi = t_confidence_interval_list(maes)
    rmse_mean, rmse_lo, rmse_hi = t_confidence_interval_list(rmses)
    da_mean, da_lo, da_hi = t_confidence_interval_list(das)

    out = {
        "MAE_mean": mae_mean, "MAE_lo": mae_lo, "MAE_hi": mae_hi,
        "RMSE_mean": rmse_mean, "RMSE_lo": rmse_lo, "RMSE_hi": rmse_hi,
        "DA_mean": da_mean, "DA_lo": da_lo, "DA_hi": da_hi
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved metrics to", args.out)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

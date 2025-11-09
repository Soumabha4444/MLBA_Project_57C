
#!/usr/bin/env python3
"""
eval.py

Flexible evaluator for prediction JSON outputs. Supports multiple input formats:

1) New training script format:
   {
     "preds": [..],
     "y_test": [..],
     "metrics": { optional precomputed metrics }
   }

2) Older "per_fold" format:
   { "per_fold": [ { "preds": [...], "y_test": [...] }, ... ] }

3) A plain list of fold records:
   [ { "preds": [...], "y_test": [...] }, ... ]

This script computes:
 - MAE, RMSE, Directional Accuracy (DA)
 - 95% CI via bootstrap (default 1000 samples)
 - For multiple folds, returns mean + 2.5/97.5 percentiles across folds.

Writes JSON to --out path.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------
# Metrics / utils
# ---------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    # directional accuracy: equality of sign (zero treated as sign 0)
    sign_equal = (np.sign(y_pred) == np.sign(y_true))
    da = 100.0 * float(np.mean(sign_equal))
    return {"MAE": mae, "RMSE": rmse, "DA": da}


def bootstrap_sample_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Dict[str, Tuple[float, float, float]]:
    """
    For a single fold: sample indices with replacement and compute bootstrap distribution
    Returns dict: key -> (mean, lo, hi)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n == 0:
        raise ValueError("Empty test set for bootstrap.")
    stats = {"MAE": [], "RMSE": [], "DA": []}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        m = compute_metrics(yt, yp)
        stats["MAE"].append(m["MAE"])
        stats["RMSE"].append(m["RMSE"])
        stats["DA"].append(m["DA"])
    out = {}
    for k, arr in stats.items():
        arr = np.array(arr)
        out[k] = (float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return out


def aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Tuple[float, float, float]]:
    """
    Given a list of per-fold metrics (dicts with keys MAE, RMSE, DA),
    return (mean, lo, hi) where lo/hi are 2.5/97.5 percentiles across folds.
    """
    arrs = {"MAE": [], "RMSE": [], "DA": []}
    for fm in fold_metrics:
        arrs["MAE"].append(fm["MAE"])
        arrs["RMSE"].append(fm["RMSE"])
        arrs["DA"].append(fm["DA"])
    out = {}
    for k, vals in arrs.items():
        vals = np.array(vals)
        out[k] = (float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return out


# ---------------------
# Loader helpers
# ---------------------
def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def ensure_array(x) -> np.ndarray:
    """
    Convert list-like to numpy array. Handles nested lists etc.
    """
    return np.asarray(x, dtype=float)


def detect_preds_structure(data: Any) -> Tuple[str, Any]:
    """
    Return a tuple (kind, payload)
    kind: "single", "folds", "legacy_per_fold"
    payload: for "single" -> dict with 'preds' and 'y_test'
             for "folds" -> list of per-fold dicts (each with preds,y_test)
    """
    if isinstance(data, dict):
        if "preds" in data and "y_test" in data:
            return "single", {"preds": data["preds"], "y_test": data["y_test"], "metrics": data.get("metrics", None)}
        if "per_fold" in data and isinstance(data["per_fold"], list):
            return "folds", data["per_fold"]
        # sometimes authors put folds under 'folds' or 'results'
        for k in ("folds", "results", "perFold"):
            if k in data and isinstance(data[k], list):
                return "folds", data[k]
        # If dict of many entries that look like fold keys, attempt to coerce
        # Otherwise fail and try list detection below
    if isinstance(data, list):
        # list of fold-like objects? ensure each has preds & y_test
        if len(data) > 0 and all(isinstance(x, dict) and "preds" in x and "y_test" in x for x in data):
            return "folds", data
    raise ValueError("Unrecognized preds_json structure. Expect single dict with 'preds' and 'y_test', or a list/dict containing per-fold entries.")


# ---------------------
# Main
# ---------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds_json", required=True, help="Predictions JSON created by training script")
    p.add_argument("--out", required=True, help="Output metrics JSON path")
    p.add_argument("--n_boot", type=int, default=1000, help="Bootstrap samples when computing CIs for a single-fold preds file")
    p.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed")
    args = p.parse_args()

    preds_path = Path(args.preds_json)
    if not preds_path.exists():
        raise FileNotFoundError(args.preds_json)

    data = load_json(preds_path)

    kind, payload = detect_preds_structure(data)

    out_metrics: Dict[str, Any] = {}
    out_payload = {"source": str(preds_path), "input_kind": kind}

    if kind == "single":
        preds = ensure_array(payload["preds"])
        y_test = ensure_array(payload["y_test"])
        if preds.shape != y_test.shape:
            # try to flatten if preds nested
            preds = preds.flatten()
            y_test = y_test.flatten()
            if preds.shape != y_test.shape:
                raise ValueError(f"preds and y_test shapes mismatch after flatten: {preds.shape} vs {y_test.shape}")
        # compute point metrics
        m = compute_metrics(y_test, preds)
        # bootstrap per-sample for CI
        boot = bootstrap_sample_metrics(y_test, preds, n_boot=args.n_boot, seed=args.seed)
        out_metrics["MAE_mean"], out_metrics["MAE_lo"], out_metrics["MAE_hi"] = boot["MAE"]
        out_metrics["RMSE_mean"], out_metrics["RMSE_lo"], out_metrics["RMSE_hi"] = boot["RMSE"]
        out_metrics["DA_mean"], out_metrics["DA_lo"], out_metrics["DA_hi"] = boot["DA"]
        out_metrics["n_test"] = int(len(y_test))
        out_metrics["bootstrap_samples"] = int(args.n_boot)
        out_metrics["bootstrap_seed"] = int(args.seed)
        # include training-provided metrics if present
        if payload.get("metrics") is not None:
            out_payload["provided_metrics"] = payload["metrics"]
    elif kind == "folds":
        folds = payload  # list of dicts
        fold_metrics = []
        n_total = 0
        for f in folds:
            if not ("preds" in f and "y_test" in f):
                raise ValueError("Each fold entry must contain 'preds' and 'y_test'.")
            preds = ensure_array(f["preds"]).flatten()
            y_test = ensure_array(f["y_test"]).flatten()
            if preds.shape != y_test.shape:
                raise ValueError("preds and y_test must have same shape within each fold.")
            n_total += len(y_test)
            m = compute_metrics(y_test, preds)
            fold_metrics.append(m)
        # aggregate metrics across folds (mean and percentile)
        agg = aggregate_fold_metrics(fold_metrics)
        out_metrics["MAE_mean"], out_metrics["MAE_lo"], out_metrics["MAE_hi"] = agg["MAE"]
        out_metrics["RMSE_mean"], out_metrics["RMSE_lo"], out_metrics["RMSE_hi"] = agg["RMSE"]
        out_metrics["DA_mean"], out_metrics["DA_lo"], out_metrics["DA_hi"] = agg["DA"]
        out_metrics["n_folds"] = len(fold_metrics)
        out_metrics["n_test_total"] = int(n_total)
    else:
        raise RuntimeError("unexpected structure type")

    # Write out combined JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": out_metrics, "meta": out_payload}, f, indent=2)

    print(f"Wrote metrics to {out_path}")
    print(json.dumps(out_metrics, indent=2))


if __name__ == "__main__":
    main()

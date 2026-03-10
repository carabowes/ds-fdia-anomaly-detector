import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.ml.detectors.isolation_forest import IsolationForestDetector
from src.ml.detectors.one_class_svm import OneClassSVMDetector
from src.ml.detectors.local_outlier_factor import LOFDetector


def load_features_jsonl(path: Path, key: str = "features") -> Tuple[np.ndarray, np.ndarray]:
    ts: List[int] = []
    feats: List[List[float]] = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            ts.append(int(rec["t"]))
            feats.append(rec[key])
    return np.asarray(ts), np.asarray(feats, dtype=float)


def make_windows(F: np.ndarray, W: int) -> np.ndarray:
    T, d = F.shape
    if T < W:
        raise ValueError(f"Not enough timesteps ({T}) for window_size={W}")
    X = np.empty((T - W + 1, W * d), dtype=float)
    for i in range(T - W + 1):
        X[i] = F[i : i + W].reshape(-1)
    return X


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)

def calibrate_threshold(detector, X_clean, q):
    """
    Compute anomaly scores on clean data and set threshold at quantile q.
    """
    out = detector.predict(X_clean)
    scores = np.asarray(out["scores"], dtype=float)
    threshold = np.percentile(scores, q)
    detector.threshold_ = threshold
    return detector

def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path, nargs = "+")
    p.add_argument("--window_size", type=int, default=5)
    p.add_argument("--out_dir", type=Path, default=Path("trained_detectors_streaming"))
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--threshold_quantile", type=float, default=99.0)

    p.add_argument("--ocsvm_nu", type=float, default=0.01)
    p.add_argument("--ocsvm_gamma", type=str, default="scale")

    p.add_argument("--iforest_n_estimators", type=int, default=200)
    p.add_argument("--iforest_contamination", type=float, default=0.01)

    p.add_argument("--lof_n_neighbors", type=int, default=35)
    p.add_argument("--lof_contamination", type=float, default=0.01)

    p.add_argument(
        "--representation",
        type=str,
        choices=["innovations", "residuals", "state_derivative", "state"],
        default="innovations",
    )

    args = p.parse_args()
    rep = args.representation

    # feat_path = args.run_dir / "features.jsonl"
    # if not feat_path.exists():
    #     raise FileNotFoundError(feat_path)

    # _, F = load_features_jsonl(feat_path)
    # X = make_windows(F, args.window_size)
    all_windows = []

    for run_dir in args.run_dir:
        feat_path = run_dir / "features.jsonl"
        if not feat_path.exists():
            raise FileNotFoundError(feat_path)

        _, F = load_features_jsonl(feat_path)
        X_run = make_windows(F, args.window_size)
        all_windows.append(X_run)

    X = np.vstack(all_windows)

    print(f"Training windows: {X.shape}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(
        scaler,
        args.out_dir / f"scaler_ieee9_W{args.window_size}_{rep}.pkl",
    )

    clean_mask = np.ones(Xs.shape[0], dtype=int)

    # ---------------- OCSVM ----------------
    ocsvm = OneClassSVMDetector(
        nu=args.ocsvm_nu,
        gamma=args.ocsvm_gamma,
    )
    ocsvm.fit(Xs, clean_mask=clean_mask)
    ocsvm = calibrate_threshold(ocsvm, Xs, args.threshold_quantile)

    save_pickle(
        ocsvm,
        args.out_dir / f"ocsvm_ieee9_W{args.window_size}_{rep}.pkl",
    )

    # ---------------- LOF ----------------
    lof = LOFDetector(
        n_neighbors=args.lof_n_neighbors,
        contamination=args.lof_contamination,
    )
    lof.fit(Xs, clean_mask=clean_mask)
    lof = calibrate_threshold(lof, Xs, args.threshold_quantile)

    save_pickle(
        lof,
        args.out_dir / f"lof_ieee9_W{args.window_size}_{rep}.pkl",
    )

    # ---------------- Isolation Forest ----------------
    iforest = IsolationForestDetector(
        n_estimators=args.iforest_n_estimators,
        contamination=args.iforest_contamination,
        random_state=args.seed,
    )
    iforest.fit(Xs, clean_mask=clean_mask)
    iforest = calibrate_threshold(iforest, Xs, args.threshold_quantile)

    save_pickle(
        iforest,
        args.out_dir / f"iforest_ieee9_W{args.window_size}_{rep}.pkl",
    )

    print("Saved calibrated streaming-trained detectors to:", args.out_dir)


if __name__ == "__main__":
    main()
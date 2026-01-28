from pathlib import Path
import numpy as np
import argparse

from src.datasets.windowed_dataset import build_windowed_dataset
from src.ml.detectors.isolation_forest import IsolationForestDetector
from src.ml.detectors.one_class_svm import OneClassSVMDetector
from src.ml.detectors.local_outlier_factor import LOFDetector
from src.ml.alarm_projection import window_alarms_to_timesteps
from src.io.load_pipeline_run import load_pipeline_run
from src.ml.evaluation import evaluate_timestep_detection
from src.datasets.windowed_dataset import compute_clean_window_mask

"""
Entry point for anomaly detection and evaluation on exported pipeline runs.

This script loads a previously generated pipeline run, constructs windowed datasets, trains an unsupervised anomaly detector 
on clean data only, projects window-level alarms to timesteps, and evaluates detection performance against ground-truth attack labels.
"""

# LIVE_MODE = True
# COMMISSIONING_T = 200
# TRIM_FRACTION = 0.2 # discard top 20% most anomolous
# MIN_TRAIN_WINDOWS = 50 #safety guard

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid search detection experiments for IEEE-9 FDIA"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["standard", "random", "stealth"],
        default="standard",
        help="FDIA scenario to evaluate",
    )

    parser.add_argument(
        "--detector",
        type=str,
        choices=["isolation_forest", "ocsvm", "lof"],
        default="isolation_forest",
        help="Anomaly detector to use",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed of the pipeline run",
    )

    parser.add_argument(
        "--T",
        type=int,
        default=200,
        help="No. of timesteps in the pipeline run",
    )

    # parser.add_argument("--k", type=int, default=1, help="k alarms required")

    # parser.add_argument("--m", type=int, default=1, help="window length (timesteps)")
    
    return parser.parse_args()

def apply_k_out_of_m(timestep_alarms: np.ndarray, k: int, m: int) -> np.ndarray:
    # Returns binary array where an alarm triggers at t if >=k  alarms occured in [t-m+1, t]
    alarms = timestep_alarms.astype(int)
    if m <= 1 or k <= 1:
        return alarms.astype(bool)
    window_sum = np.convolve(alarms, np.ones(m, dtype=int), mode="full")[: len(alarms)]
    return (window_sum >= k)

# def commissioning_train_mask(start_idx: np.ndarray, window_size: int, commissioning_T: int) -> np.ndarray:
#     # window must be fully contained in commissioning region
#     return (start_idx + window_size) <= commissioning_T

# def robust_trim_mask(X: np.ndarray, base_mask: np.ndarray, detector, trim_fraction: float) -> np.ndarray:
#     """
#     Unsupervised robust calibration:
#     - fit on base_mask windows
#     - score those windows
#     - keep the least-anomalous (1-trim_fraction) fraction
#     """
#     idx = np.where(base_mask)[0]
#     if idx.size < MIN_TRAIN_WINDOWS or trim_fraction <= 0.0:
#         return base_mask

#     # provisional fit on base windows only (no clean_mask needed because we slice X)
#     detector.fit(X[idx])

#     scores = detector.score(X[idx])  # higher = more anomalous (your convention)
#     cutoff = np.percentile(scores, 100.0 * (1.0 - trim_fraction))
#     keep = scores <= cutoff

#     trimmed = np.zeros_like(base_mask, dtype=bool)
#     trimmed[idx[keep]] = True

#     # safety guard
#     if trimmed.sum() < MIN_TRAIN_WINDOWS:
#         return base_mask
#     return trimmed

def main():
    args = parse_args()
    RUN_DIR = ( Path("runs")
        / "ieee9"
        / args.scenario
        / f"T_{args.T}"
        / f"seed_{args.seed}"
        )
    # WINDOW_SIZE = 10
    STRIDE = 1
    # CALIBRATION_T = 100
    detector_name = args.detector

    REPRESENTATIONS = ["residuals", "innovations"]

    # Hyperparameter grids for isolation forest
    threshold_grid = [90, 95, 97.5, 99, 99.5]
    n_estimators_grid = [100, 200]
    window_size_grid = [5, 10, 20]

    innovation_alpha_grid = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    # OCSVM hyperparameters
    # nu_grid = [0.001, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.125, 0.15]
    # nu_grid = [0.01, 0.02, 0.05]
    # gamma_grid = ["scale"]
    gamma_grid = ["scale", "auto", 0.01, 0.1, 1.0]
    nu_grid = [0.001, 0.005, 0.01, 0.02, 0.05]
    # gamma_grid = ["scale", "auto"]
    #LOF
    lof_neighbors_grid = [10, 20, 30, 40]


    best_result = None
    results = []
    
    print("FDIA Detection Grid Search")
    print(f"Scenario : {args.scenario}")
    print(f"Seed     : {args.seed}")
    # print(f"LIVE_MODE: {LIVE_MODE} (COMMISSIONING_T={COMMISSIONING_T}, TRIM={TRIM_FRACTION})")

    for rep in REPRESENTATIONS:

        print("\n" + "=" * 90)
        print(f"REPRESENTATION = {rep}")
        print("=" * 90)

        alpha_grid = (
            innovation_alpha_grid if rep == "innovations" else [None]
        )
        for window_size in window_size_grid:
            for alpha in alpha_grid:
                # Build dataset
                X, window_metadata, attack_mask = build_windowed_dataset(
                    run_dir=RUN_DIR,
                    window_size=window_size,
                    stride=STRIDE,
                    representation=rep,
                    innovation_alpha = alpha if alpha is not None else 0.3,
                )
                # print("DEBUG X shape:", X.shape)
                window_clean_mask = compute_clean_window_mask(
                    attack_mask=attack_mask,
                    window_starts=window_metadata["start_indices"],
                    window_size=window_size,
                )

                T = len(attack_mask)

                # start_idx = np.asarray(window_metadata["start_indices"], dtype=int)
                # T = len(attack_mask)

                # if LIVE_MODE:
                #     train_base = commissioning_train_mask(start_idx, window_size, COMMISSIONING_T)
                # else:
                #     train_base = compute_clean_window_mask(
                #         attack_mask=attack_mask,
                #         window_starts=start_idx,
                #         window_size=window_size,
                #     ).astype(bool)
            
                if detector_name == "isolation_forest":
                    for n_estimators in n_estimators_grid:
                        for q in threshold_grid:
                            detector = IsolationForestDetector(
                                n_estimators = n_estimators,
                                threshold_quantile= q,
                                random_state= 42,
                            )
                            detector.fit(X, clean_mask=window_clean_mask)
                            out = detector.predict(X)
                            window_alarms = out["alarms"]

                            timestep_alarms = window_alarms_to_timesteps(
                            window_alarms=window_alarms,
                            start_indices=window_metadata["start_indices"],
                            window_size=window_size,
                            T=T,
                        )
                            metrics = evaluate_timestep_detection(
                            attack_mask=attack_mask,
                            timestep_alarms=timestep_alarms,
                        )
                            # train_mask = robust_trim_mask(X, train_base, detector, TRIM_FRACTION) if LIVE_MODE else train_base
                            # detector.fit(X, clean_mask=train_mask.astype(int))
                            
                            # out = detector.predict(X)
                            # # window_alarms = out["alarms"]

                            # timestep_alarms = window_alarms_to_timesteps(
                            #     window_alarms=out["alarms"],
                            #     start_indices=start_idx,
                            #     window_size=window_size,
                            #     T=T,
                            # )
                        #     metrics = evaluate_timestep_detection(
                        #     attack_mask=attack_mask,
                        #     timestep_alarms=timestep_alarms,
                        # )
                        #     mitigation_alarms = apply_k_out_of_m(timestep_alarms, k=args.k, m=args.m)
                        #     # Metrics reflect what mitigation system would do
                        #     metrics = evaluate_timestep_detection(
                        #     attack_mask=attack_mask,
                        #     timestep_alarms=mitigation_alarms,
                        # ) 

                            result = {
                                "detector": "isolation_forest",
                                "representation": rep,
                                "n_estimators": n_estimators,
                                "window_size": window_size,
                                "threshold_q": q,
                                "alpha": alpha if rep == "innovations" else None,
                                **metrics,
                            }
                            results.append(result)
                            # if rep == "innovations":
                            #     print("DEBUG added innovations result:", result["detector"], "W", window_size, "q", q,
                            #         "TPR", metrics["TPR"], "FPR", metrics["FPR"])
                            if best_result is None:
                                best_result = result
                            else:
                                if (
                                    result["f1"] > best_result["f1"]
                                    or (
                                        result["f1"] == best_result["f1"]
                                        and result["FPR"] < best_result["FPR"]
                                    )
                                ):
                                    best_result = result

                elif detector_name == "ocsvm":
                    for nu in nu_grid:
                        for gamma in gamma_grid:
                            for q in threshold_grid:
                                detector = OneClassSVMDetector(
                                    nu=nu,
                                    gamma = gamma,
                                    threshold_quantile= q,
                                )
                                detector.fit(X, clean_mask=window_clean_mask)
                                out = detector.predict(X)
                                window_alarms = out["alarms"]

                                timestep_alarms = window_alarms_to_timesteps(
                                window_alarms=window_alarms,
                                start_indices=window_metadata["start_indices"],
                                window_size=window_size,
                                T=T,
                            )
                                metrics = evaluate_timestep_detection(
                                attack_mask=attack_mask,
                                timestep_alarms=timestep_alarms,
                        )

                            #     train_mask = robust_trim_mask(X, train_base, detector, TRIM_FRACTION) if LIVE_MODE else train_base
                            #     detector.fit(X, clean_mask=train_mask.astype(int))
                                
                            #     out = detector.predict(X)
                            #     timestep_alarms = window_alarms_to_timesteps(
                            #         window_alarms=out["alarms"],
                            #         start_indices=start_idx,
                            #         window_size=window_size,
                            #         T=T,
                            #     )
                           
                            #     mitigation_alarms = apply_k_out_of_m(timestep_alarms, k=args.k, m=args.m)
                            #     # Metrics reflect what mitigation system would do
                            #     metrics = evaluate_timestep_detection(
                            #     attack_mask=attack_mask,
                            #     timestep_alarms=mitigation_alarms,
                            # ) 
                            
                                result = {
                                    "detector": "ocsvm",
                                    "representation": rep,
                                    "window_size": window_size,
                                    "threshold_q": q,
                                    "nu": nu,
                                    "gamma": gamma,
                                    "alpha": alpha if rep == "innovations" else None,
                                    **metrics,
                                }
                                results.append(result)
                                if best_result is None:
                                    best_result = result
                                else:
                                    if (
                                        result["f1"] > best_result["f1"]
                                        or (
                                            result["f1"] == best_result["f1"]
                                            and result["FPR"] < best_result["FPR"]
                                        )
                                    ):
                                        best_result = result

                elif detector_name == "lof":
                    for n_neighbors in lof_neighbors_grid:
                        for q in threshold_grid:
                            detector = LOFDetector(
                                n_neighbors=n_neighbors,
                                threshold_quantile=q,
                            )

                            detector.fit(X, clean_mask=window_clean_mask)
                            out = detector.predict(X)
                            window_alarms = out["alarms"]

                            timestep_alarms = window_alarms_to_timesteps(
                            window_alarms=window_alarms,
                            start_indices=window_metadata["start_indices"],
                            window_size=window_size,
                            T=T,
                        )
                            metrics = evaluate_timestep_detection(
                            attack_mask=attack_mask,
                            timestep_alarms=timestep_alarms,
                        )
                        #     train_mask = robust_trim_mask(X, train_base, detector, TRIM_FRACTION) if LIVE_MODE else train_base
                        #     detector.fit(X, clean_mask=train_mask.astype(int))
                            
                        #     out = detector.predict(X)
                        #     # window_alarms = out["alarms"]

                        #     timestep_alarms = window_alarms_to_timesteps(
                        #         window_alarms=out["alarms"],
                        #         start_indices=start_idx,
                        #         window_size=window_size,
                        #         T=T,
                        #     )
                    
                        #     mitigation_alarms = apply_k_out_of_m(timestep_alarms, k=args.k, m=args.m)
                        #     # Metrics reflect what mitigation system would do
                        #     metrics = evaluate_timestep_detection(
                        #     attack_mask=attack_mask,
                        #     timestep_alarms=mitigation_alarms,
                        # ) 

                            result = {
                                "detector": "lof",
                                "representation": rep,
                                "window_size": window_size,
                                "threshold_q": q,
                                "n_neighbors": n_neighbors,
                                "alpha": alpha if rep == "innovations" else None,
                                **metrics,
                            }

                            results.append(result)
                            if best_result is None:
                                best_result = result
                            else:
                                if (
                                    result["f1"] > best_result["f1"]
                                    or (
                                        result["f1"] == best_result["f1"]
                                        and result["FPR"] < best_result["FPR"]
                                    )
                                ):
                                    best_result = result
                           
                print("USING DETECTOR:", type(detector).__name__)

    print("\n BEST CONFIGURATION")
    print(
        f"representation: {best_result['representation']}\n"
        f"window_size: {best_result['window_size']}\n"
        f"threshold_q: {best_result['threshold_q']}\n"
        f"alpha: {best_result['alpha']}\n"
        f"gamma: {best_result.get('gamma', 'N/A')}\n"
        f"nu: {best_result.get('nu', 'N/A')}\n"
        f"lof_neighbours: {best_result.get('n_neighbors', 'N/A')}\n"
    )

    print("\nEvaluation metrics (BEST)")
    for k in ["TP", "FP", "FN", "TN", "TPR", "FPR", "precision", "recall", "f1", "accuracy", "detection_delay"]:
        print(f"{k}: {best_result[k]}")

if __name__ == "__main__":
    main()
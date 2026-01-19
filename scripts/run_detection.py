from pathlib import Path
import numpy as np
import argparse

from src.datasets.windowed_dataset import build_windowed_dataset
from src.ml.detectors.isolation_forest import IsolationForestDetector
from src.ml.alarm_projection import window_alarms_to_timesteps
from src.io.load_pipeline_run import load_pipeline_run
from src.ml.evaluation import evaluate_timestep_detection
from src.datasets.windowed_dataset import compute_clean_window_mask

"""
Entry point for anomaly detection and evaluation on exported pipeline runs.

This script loads a previously generated pipeline run, constructs windowed atasets, trains an unsupervised anomaly detector 
on clean data only, projects window-level alarms to timesteps, and evaluates detection performance against ground-truth attack labels.
"""

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
        "--seed",
        type=int,
        default=42,
        help="Random seed of the pipeline run",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    RUN_DIR = ( Path("runs")
        / "ieee9"
        / args.scenario
        / f"seed_{args.seed}"
        )
    # WINDOW_SIZE = 10
    STRIDE = 1
    REPRESENTATIONS = ["residuals", "innovations"]

    # Hyperparameter grids
    threshold_grid = [90, 95, 97.5, 99]
    n_estimators_grid = [100, 200]
    window_size_grid = [5, 10, 20]

    innovation_alpha_grid = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    best_result = None
    results = []
    
    print("FDIA Detection Grid Search")
    print(f"Scenario : {args.scenario}")
    print(f"Seed     : {args.seed}")

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

                window_clean_mask = compute_clean_window_mask(
                    attack_mask=attack_mask,
                    window_starts=window_metadata["start_indices"],
                    window_size=window_size,
                )

                T = len(attack_mask)

                for n_estimators in n_estimators_grid:
                    for q in threshold_grid:
                        detector = IsolationForestDetector(
                            n_estimators = 200,
                            threshold_quantile= 95,
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
                        
                        # print("Num window alarms:", int(window_alarms.sum()))
                        # print("Num timestep alarms:", int(timestep_alarms.sum()))
                        # print("Num attack timesteps:", int(attack_mask.sum()))

                        metrics = evaluate_timestep_detection(
                            attack_mask=attack_mask,
                            timestep_alarms=timestep_alarms,
                        )

                        result = {
                            "representation": rep,
                            # "innovation_alpha": alpha,
                            "window_size": window_size,
                            "n_estimators": n_estimators,
                            "threshold_q": q,
                            "alpha": alpha if rep == "innovations" else None,
                            **metrics,
                        }
                        
                        results.append(result)

                        # Track best result by F1
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
                        
    # print("\nGrid search results:")
    # print("-" * 80)
    # for r in results:
    #     print(
    #         f"{r['representation']:>12} | "
    #         f"W={r['window_size']:>2} | "
    #         f"trees={r['n_estimators']:>3} | "
    #         f"q={r['threshold_q']:>4} | "
    #         f"a={r['alpha']} | "
    #         f"TPR={r['TPR']:.2f} | "
    #         f"FPR={r['FPR']:.2f} | "
    #         f"precision={r['precision']:.2f} | "
    #         f"F1={r['f1']:.2f}"
    #     )

    # print("\nEvaluation metrics")
    # for k, v in metrics.items():
    #     print(f"{k}: {v}")
    
    print("\n BEST CONFIGURATION")
    print(
        f"representation: {best_result['representation']}\n"
        f"window_size: {best_result['window_size']}\n"
        f"n_estimators: {best_result['n_estimators']}\n"
        f"threshold_q: {best_result['threshold_q']}\n"
        f"alpha: {best_result['alpha']}"
    )

    print("\nEvaluation metrics (BEST)")
    for k in ["TP", "FP", "FN", "TN", "TPR", "FPR", "precision", "recall", "f1", "accuracy", "detection_delay"]:
        print(f"{k}: {best_result[k]}")

if __name__ == "__main__":
    main()
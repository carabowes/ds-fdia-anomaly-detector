from pathlib import Path
import numpy as np
import argparse
import pickle

from src.datasets.windowed_dataset import build_windowed_dataset
from src.ml.detectors.isolation_forest import IsolationForestDetector
from src.ml.detectors.one_class_svm import OneClassSVMDetector
from src.ml.detectors.local_outlier_factor import LOFDetector
from src.ml.alarm_projection import window_alarms_to_timesteps
from src.io.load_pipeline_run import load_pipeline_run
from src.ml.evaluation import evaluate_timestep_detection
from src.datasets.windowed_dataset import compute_clean_window_mask
from src.ml.mitigation_metrics import compute_false_incident_rate, extract_alarm_segments, evaluate_episode_detection, summarise_episode_detection

"""
Entry point for anomaly detection and evaluation on exported pipeline runs.

This script loads a previously generated pipeline run, constructs windowed datasets, trains an unsupervised anomaly detector 
on clean data only, projects window-level alarms to timesteps, and evaluates detection performance against ground-truth attack labels.
"""
Path("trained_detectors").mkdir(exist_ok=True)

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
    
    return parser.parse_args()

def apply_k_out_of_m(timestep_alarms: np.ndarray, k: int, m: int) -> np.ndarray:
    # Returns binary array where an alarm triggers at t if >=k  alarms occured in [t-m+1, t]
    alarms = timestep_alarms.astype(int)
    if m <= 1 or k <= 1:
        return alarms.astype(bool)
    window_sum = np.convolve(alarms, np.ones(m, dtype=int), mode="full")[: len(alarms)]
    return (window_sum >= k)

def main():
    args = parse_args()
    RUN_DIR = ( Path("runs")
        / "ieee9"
        / args.scenario
        / f"T_{args.T}"
        / f"seed_{args.seed}"
        )

    # Load attack episodes from the pipeline run
    run = load_pipeline_run(RUN_DIR)
    print(run.keys())
    for k,v in run.items():
        try:
            print(k, v.shape)
        except:
            print(k, type(v))
    attack_episodes = run["metadata"]["attack_episodes"]
    episode_intervals = [(ep["start"], ep["end"]) for ep in attack_episodes]
    
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
    
    best_models = {
    "isolation_forest": {"score": -np.inf, "params": None},
    "ocsvm": {"score": -np.inf, "params": None},
    "lof": {"score": -np.inf, "params": None},
}

    best_result = None
    best_params = None
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
                            # --- Mitigation-level metrics (incident / episode based) ---

                            alarm_segments = extract_alarm_segments(timestep_alarms)

                            false_incident_metrics = compute_false_incident_rate(
                                alarm_segments=alarm_segments,
                                attack_episodes=episode_intervals,
                                T=T,
                            )

                            episode_results = evaluate_episode_detection(
                                alarm_segments=alarm_segments,
                                attack_episodes=episode_intervals,
                            )

                            episode_summary = summarise_episode_detection(episode_results)

                            result = {
                                "detector": "isolation_forest",
                                "representation": rep,
                                "n_estimators": n_estimators,
                                "window_size": window_size,
                                "threshold_q": q,
                                "alpha": alpha if rep == "innovations" else None,
                                "false_incidents_per_500": false_incident_metrics["false_incidents_per_500"],
                                "num_false_incidents": false_incident_metrics["false_incidents"],
                                "episode_detection_rate": episode_summary["detection_rate"],
                                "median_ttfd": episode_summary["median_ttfd"],
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
                                    best_params = {
                                        "detector": "isolation_forest",
                                        "representation": rep,
                                        "window_size": window_size,
                                        "alpha": alpha,
                                        "n_estimators": n_estimators,
                                        "threshold_q": q,
                                    }

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
                                # --- Mitigation-level metrics (incident / episode based) ---

                                alarm_segments = extract_alarm_segments(timestep_alarms)

                                false_incident_metrics = compute_false_incident_rate(
                                    alarm_segments=alarm_segments,
                                    attack_episodes=episode_intervals,
                                    T=T,
                                )

                                episode_results = evaluate_episode_detection(
                                    alarm_segments=alarm_segments,
                                    attack_episodes=episode_intervals,
                                )

                                episode_summary = summarise_episode_detection(episode_results)
                            
                                result = {
                                    "detector": "ocsvm",
                                    "representation": rep,
                                    "window_size": window_size,
                                    "threshold_q": q,
                                    "nu": nu,
                                    "gamma": gamma,
                                    "alpha": alpha if rep == "innovations" else None,
                                    "false_incidents_per_500": false_incident_metrics["false_incidents_per_500"],
                                    "num_false_incidents": false_incident_metrics["false_incidents"],
                                    "episode_detection_rate": episode_summary["detection_rate"],
                                    "median_ttfd": episode_summary["median_ttfd"],
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
                                        best_params = {
                                            "detector": "ocsvm",
                                            "representation": rep,
                                            "window_size": window_size,
                                            "alpha": alpha,
                                            "nu": nu,
                                            "gamma": gamma,
                                            "threshold_q": q,
                                        }
                                
                                # is_better = (
                                #     best_result is None
                                #     or result["f1"] > best_result["f1"]
                                #     or (
                                #         result["f1"] == best_result["f1"]
                                #         and result["false_incidents_per_500"] < best_result["false_incidents_per_500"]
                                #     )
                                # )

                                # if is_better:
                                #     print(
                                #         f"[MITIGATION BEST] "
                                #         f"false_incidents={false_incident_metrics['false_incidents']}, "
                                #         f"per500={false_incident_metrics['false_incidents_per_500']:.2f}, "
                                #         f"episode_detection_rate={episode_summary['detection_rate']:.2f}, "
                                #         f"median_ttfd={episode_summary['median_ttfd']}"
                                #     )

    #                             results.append(result)
    #                             if best_result is None:
    #                                 best_result = result
    #                             else:
    #                                 if (
    #                                     result["f1"] > best_result["f1"]
    #                                     or (
    #                                         result["f1"] == best_result["f1"]
    #                                         and result["FPR"] < best_result["FPR"]
    #                                     )
    #                                 ):
    #                                     best_result = result
    #                             is_better = (
    #                                 best_result is None
    #                                 or result["f1"] > best_result["f1"]
    #                                 or (
    #                                     result["f1"] == best_result["f1"]
    #                                     and result["false_incidents_per_500"] < best_result["false_incidents_per_500"]
    #                                 )
    #                             )

    #                             if is_better:
    #                                 print(
    #                                     f"[MITIGATION BEST] "
    #                                     f"false_incidents={false_incident_metrics['false_incidents']}, "
    #                                     f"per500={false_incident_metrics['false_incidents_per_500']:.2f}, "
    #                                     f"episode_detection_rate={episode_summary['detection_rate']:.2f}, "
    #                                     f"median_ttfd={episode_summary['median_ttfd']}"
    # )         

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
                        # --- Mitigation-level metrics (incident / episode based) ---

                            alarm_segments = extract_alarm_segments(timestep_alarms)

                            false_incident_metrics = compute_false_incident_rate(
                                alarm_segments=alarm_segments,
                                attack_episodes=episode_intervals,
                                T=T,
                            )

                            episode_results = evaluate_episode_detection(
                                alarm_segments=alarm_segments,
                                attack_episodes=episode_intervals,
                            )

                            episode_summary = summarise_episode_detection(episode_results)
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
                                "false_incidents_per_500": false_incident_metrics["false_incidents_per_500"],
                                "num_false_incidents": false_incident_metrics["false_incidents"],
                                "episode_detection_rate": episode_summary["detection_rate"],
                                "median_ttfd": episode_summary["median_ttfd"],
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
                                    best_params = {
                                        "detector": "lof",
                                        "representation": rep,
                                        "window_size": window_size,
                                        "alpha": alpha,
                                        "n_neighbors": n_neighbors,
                                        "threshold_q": q,
                                    }
                            
                # print("USING DETECTOR:", type(detector).__name__)
    # Rebuild dataset for BEST configuration
    X_best, window_metadata_best, attack_mask_best = build_windowed_dataset(
        run_dir=RUN_DIR,
        window_size=best_params["window_size"],
        stride=STRIDE,
        representation=best_params["representation"],
        innovation_alpha=best_params["alpha"] if best_params["representation"] == "innovations" else 0.3,
    )

    window_clean_mask_best = compute_clean_window_mask(
        attack_mask=attack_mask_best,
        window_starts=window_metadata_best["start_indices"],
        window_size=best_params["window_size"],
    )

    # Train final detector
    if best_params["detector"] == "isolation_forest":
        best_detector = IsolationForestDetector(
            n_estimators=best_params["n_estimators"],
            threshold_quantile=best_params["threshold_q"],
            random_state=42,
        )
        save_path = "trained_detectors/iforest_ieee9.pkl"

    elif best_params["detector"] == "ocsvm":
        best_detector = OneClassSVMDetector(
            nu=best_params["nu"],
            gamma=best_params["gamma"],
            threshold_quantile=best_params["threshold_q"],
        )
        save_path = "trained_detectors/ocsvm_ieee9.pkl"

    elif best_params["detector"] == "lof":
        best_detector = LOFDetector(
            n_neighbors=best_params["n_neighbors"],
            threshold_quantile=best_params["threshold_q"],
        )
        save_path = "trained_detectors/lof_ieee9.pkl"

    else:
        raise ValueError(f"Unknown detector type: {best_params['detector']}")

    # Fit on CLEAN windows only
    best_detector.fit(X_best, clean_mask=window_clean_mask_best)

    # Persist detector
    with open(save_path, "wb") as f:
        pickle.dump(best_detector, f)

    print(f"[SAVED BEST] {save_path}")

    print("BEST CONFIG:", best_result)
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
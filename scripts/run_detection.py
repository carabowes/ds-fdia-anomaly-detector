from pathlib import Path
import numpy as np

from src.datasets.windowed_dataset import build_windowed_dataset
from src.ml.detectors.isolation_forest import IsolationForestDetector
from src.ml.alarm_projection import window_alarms_to_timesteps
from src.io.load_pipeline_run import load_pipeline_run
from src.ml.evaluation import evaluate_timestep_detection
from src.datasets.windowed_dataset import compute_clean_window_mask

RUN_DIR = Path("runs/ieee9/stealth/seed_42")
# WINDOW_SIZE = 10
STRIDE = 1

# X, window_metadata, attack_mask = build_windowed_dataset(
#     run_dir= RUN_DIR,
#     # window_size=WINDOW_SIZE,
#     stride=STRIDE,
#     representation="residuals",
# )

# window_clean_mask = compute_clean_window_mask(
#     attack_mask=attack_mask,
#     window_starts= window_metadata["start_indices"],
#     # window_size=WINDOW_SIZE,
# )

# T = len(attack_mask)

# Hyperparameter grid
threshold_grid = [90, 95, 97.5, 99]
n_estimators_grid = [100, 200]
window_size_grid = [5, 10, 20]
results = []

for window_size in window_size_grid:

    # Rebuild dataset for this window size
    X, window_metadata, attack_mask = build_windowed_dataset(
        run_dir=RUN_DIR,
        window_size=window_size,
        stride=STRIDE,
        representation="residuals",
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
            n_estimators=n_estimators,
            threshold_quantile=q,
            random_state=42,
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

        results.append({
            "window_size": window_size,
            "n_estimators": n_estimators,
            "threshold_q": q,
            **metrics
        })


# detector = IsolationForestDetector(random_state=42)
# detector.fit(X, clean_mask=window_clean_mask)

# out = detector.predict(X)
# window_alarms = out["alarms"]

# # Project to timesteps
# timestep_alarms = window_alarms_to_timesteps(
#     window_alarms=window_alarms,
#     start_indices=window_metadata["start_indices"],
#     window_size=WINDOW_SIZE,
#     T=T,
# )

print("\nGrid search results:")
print("-" * 80)
for r in results:
    print(
        f"trees={r['n_estimators']:3d} | "
        f"q={r['threshold_q']:>4} | "
        f"TPR={r['TPR']:.2f} | "
        f"FPR={r['FPR']:.2f} | "
        f"F1={r['f1']:.2f} | "
        f"delay={r['detection_delay']}"
    )

print("Num window alarms:", window_alarms.sum())
print("Num timestep alarms:", timestep_alarms.sum())
print("Num attack timesteps:", attack_mask.sum())

# Evaluation
metrics = evaluate_timestep_detection(
    attack_mask=attack_mask,
    timestep_alarms=timestep_alarms,
)

print("\nEvaluation metrics")
for k, v in metrics.items():
    print(f"{k}: {v}")
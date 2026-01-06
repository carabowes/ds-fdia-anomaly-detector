from pathlib import Path
import numpy as np

from src.datasets.windowed_dataset import build_windowed_dataset
from src.ml.detectors.isolation_forest import IsolationForestDetector
from src.ml.alarm_projection import window_alarms_to_timesteps
from src.io.load_pipeline_run import load_pipeline_run

RUN_DIR = Path("runs/ieee9/standard/seed_42")
WINDOW_SIZE = 10
STRIDE = 1

X, window_metadata, attack_mask = build_windowed_dataset(
    run_dir= RUN_DIR,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    representation="residuals",
)

T = len(attack_mask)

detector = IsolationForestDetector(random_state=42)
detector.fit(X)

out = detector.predict(X)
window_alarms = out["alarms"]

# Project to timesteps
timestep_alarms = window_alarms_to_timesteps(
    window_alarms=window_alarms,
    start_indices=window_metadata["start_indices"],
    window_size=WINDOW_SIZE,
    T=T,
)

print("Num window alarms:", window_alarms.sum())
print("Num timestep alarms:", timestep_alarms.sum())
print("Num attack timesteps:", attack_mask.sum())
from pathlib import Path
import json
import pandas as pd

"""
Reads the timestep-level dataset produced by the pipeline (measurements, residuals,
attack mask, convergence mask, and metadata) and returns them in a structured
dictionary for windowing and anomaly detection.
"""

def load_pipeline_run(run_dir: Path) -> dict:
    return {
        "measurements": pd.read_csv(run_dir / "measurements.csv"),
        "residuals": pd.read_csv(run_dir / "residuals.csv"),
        "attack_mask": pd.read_csv(run_dir / "attack_mask.csv"),
        "convergence_mask": pd.read_csv(run_dir / "convergence_mask.csv"),
        "metadata": json.loads((run_dir / "metadata.json").read_text()),
    }


from __future__ import annotations
from pathlib import Path
from typing import Literal, Tuple, Dict, Any
import numpy as np

from src.io.load_pipeline_run import load_pipeline_run
from src.ml.windowing import generate_sliding_windows
from src.features.innovations import compute_innovations

"""
Windowed dataset construction for anomaly detection.

Transforms timestep-level signals into fixed-size sliding windows suitable for machine learning. 
Supports multiple representations (residuals, measurements, innovations) and preserves alignment with ground-truth
attack timelines via window metadata.
"""

#Feature representation to window: residuals - residual_norm signal, measurements - raw measurement vectors Z
RepresentationType = Literal["residuals", "measurements", "innovations"] 

def build_windowed_dataset(
        run_dir: Path, # Path to a single pipeline run directory
        window_size: int, #Length of each sliding window (no of timesteps)
        stride: int, #Step size between consecutive windows
        representation: RepresentationType = "residuals", 
        innovation_alpha: float=0.3,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    
    """
    Construct a windowed dataset from a single exported pipeline run.
    This function operates on timestep-level data and does not rerun simulation, attacks or SE.
    """

    # Load exported pipeline
    data = load_pipeline_run(run_dir)

    # attack_mask is timestep-level; window-level aggregation is handled later
    attack_mask = data["attack_mask"]["attack_mask"].values
    convergence_mask = data["convergence_mask"]["converged"].values.astype(bool)

    # Select feature series
    if representation == "residuals":
        # residual norm is a 1D signal
        Z = data["residuals"]["residual_norm"].values[:, None]
        feature_dim = 1
    
    elif representation == "measurements":
        # raw measurements -> drop time coloumn
        Z = data["measurements"].drop(columns=["t"]).values
        feature_dim = Z.shape[1]
    
    elif representation == "innovations":
        # raw measurements -> drop time column
        Z_raw = data["measurements"].drop(columns=["t"]).values
        Z = compute_innovations(Z_raw, alpha=innovation_alpha)
        feature_dim = Z.shape[1]
    
    else:
        raise ValueError(f"Unsupported representation: {representation}")
    
    # Build sliding windows
    
    windows, window_metadata = generate_sliding_windows(
        Z = Z,
        window_size=window_size,
        stride=stride,
        convergence_mask=convergence_mask
    )

    # Flatten windows for ML
    num_windows, W, d = windows.shape
    X = windows.reshape(num_windows, W*d)

    # Attach metadata
    window_metadata.update(
        {
            "representation": representation,
            "window_size": window_size,
            "stride": stride,
            "feature_dim": feature_dim,
            "num_windows": int(num_windows),
            "source_run": str(run_dir),
        }
    )

    return X, window_metadata, attack_mask

def compute_clean_window_mask(
    attack_mask: np.ndarray,
    window_starts: np.ndarray,
    window_size: int,
) -> np.ndarray:
    clean_mask = np.zeros(len(window_starts), dtype=int)

    for i, s in enumerate(window_starts):
        if attack_mask[s : s + window_size].sum() == 0:
            clean_mask[i] = 1  # clean
    return clean_mask

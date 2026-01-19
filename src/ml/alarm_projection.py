from __future__ import annotations
import numpy as np
from typing import Sequence

"""
Projection of window-level alarms to timestep-level signals.

Implements a deterministic mapping from window-based anomaly decisions
to timestep-level alarm signals, enabling meaningful temporal evaluation
against ground-truth attack labels.
"""

def window_alarms_to_timesteps(
    window_alarms: np.ndarray,
    start_indices: Sequence[int],
    window_size: int,
    T: int,
) -> np.ndarray:

# Project window-level anomaly alarms to timestep-level alarms.
# A timestep is marked as anomalous if any window covering that timestep is alarmed.

    if window_alarms.ndim != 1:
        raise ValueError("window_alarms must be a 1D array")
    
    if len(window_alarms) != len(start_indices):
        raise ValueError("window_alarms and start_indices must have same length")
    
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    
    if T <= 0:
        raise ValueError("T must be positive")
    
    timestep_alarms = np.zeros(T, dtype=int)

    for alarm, start in zip(window_alarms, start_indices):
        if alarm == 1:
            end = min(start + window_size, T)
            timestep_alarms[start:end] = 1
    return timestep_alarms
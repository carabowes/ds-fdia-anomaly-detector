import numpy as np
from src.ml.windowing import generate_sliding_windows
from src.pipeline.state_estimation import run_wls_time_series

# NOTE:
# This module contains early prototype dataset builders used during exploratory experimentation.
# Current pipeline-based dataset construction operates on exported runs.

# Baseline A: Raw Measurement Windows 
def build_raw_window_dataset(
    Z: np.ndarray,
    window_size: int,
    convergence_mask: np.ndarray | None = None,
    stride: int = 1,
):
    # Build sliding window dataset from raw measurements Z

    windows, metadata = generate_sliding_windows(
        Z,
        window_size,
        convergence_mask,
        stride,
    )
    num_windows, W, d = windows.shape
    X = windows.reshape(num_windows, W * d)

    metadata["feature_type"] = "raw_measurements"
    metadata["window_size"] = window_size
    metadata["stride"] = stride
    metadata["representation"] = "flattened"

    return X, metadata

# Baseline B: Residual-based Windows
def build_residual_window_dataset(
    Z: np.ndarray,
    H: np.ndarray,
    sigma: float,
    window_size: int,
    convergence_mask: np.ndarray | None = None,
    stride: int = 1,
):
    # Build sliding window dataset from WLS residuals

    # Run WLS SE over full time series
    residual_norms, _ = run_wls_time_series(Z, H, sigma)

    # Treat residual norm as 1D signal
    residual_signal = residual_norms.reshape(-1, 1)

    windows , metadata = generate_sliding_windows(
       residual_signal,
        window_size,
        convergence_mask,
        stride,
    )
    # Flatten windows
    X = windows.reshape(windows.shape[0], window_size)

    metadata["feature_type"] = "residual_norm"
    metadata["window_size"] = window_size
    metadata["stride"] = stride
    metadata["representation"] = "flattened"

    return X, metadata

# def build_dataset(
#     Z: np.ndarray,
#     # H: np.ndarray,
#     # sigma: float,
#     window_size: int,
#     convergence_mask: np.ndarray | None = None,
#     stride: int = 1,
#     representation: str = "flattened",
# ):
#     # Build dataset from WLS residuals instead of raw measurements.
#     # Z: time series matrix of shape (T, d)
#     # window_size: number of time steps per window
#     # convergence_mask: boolean array of length T indicating convergence at each timestep
#     # stride: step size for sliding window
#     # window representation: "flattened" for now - baseline
    
#     # Run WLS over full time series
#     # R_norms, _ = run_wls_time_series(Z, H, sigma)

#     # Treat residual norm as a 1D signal
#     # R_signal = R_norms.reshape(-1, 1)

#     windows , metadata = generate_sliding_windows(
#         Z= Z,#R_signal,
#         window_size=window_size,
#         convergence_mask=convergence_mask,
#         stride=stride,
#     )

#     if representation == "flattened":
#         # Flatten each window to a 1D array
#         num_windows, W, d = windows.shape
#         X = windows.reshape(num_windows, W * d)
#     else:
#         raise ValueError(f"Unsupported representation: {representation}")
    
#     # Where X is the dataset of shape (num_windows, features)
#     # Metadata including window indices and discarded counts
#     return X, metadata
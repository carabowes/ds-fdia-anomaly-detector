import numpy as np
from typing import Literal, Optional, Tuple, Dict, Any

def generate_sliding_windows(
        Z: np.ndarray,
        window_size: int,
        convergence_mask: Optional[np.ndarray] = None,
        stride: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Basic input validation
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    if stride <= 0:
        raise ValueError("stride must be positive.")
    
    # Generate sliding windows from time-series data Z.
    T, d = Z.shape # total time steps, data dimension

    # Ensure there are enough time steps for at least one window
    if T < window_size:
        raise ValueError("Time series length T must be >= window_size.")

    # If a convergence mask is provided, check that it matches the time series length
    if convergence_mask is not None:
        if len(convergence_mask) != T:
            raise ValueError("convergence_mask length must match number of time steps T.")
   
    windows = []
    start_indices = []
    discarded = 0 # counter for discarded windows (non-converged timesteps)
     
    # Iterate over the time series using a sliding window
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        if convergence_mask is not None:
            if not np.all(convergence_mask[start:end]):
                discarded += 1
                continue
        # extract the window 
        window = Z[start:end]
        # Store valid window
        windows.append(window)
        # Record the starting timestep
        start_indices.append(start)
    
    # Convert list of windows to a numpy array
    windows = np.asarray(windows, dtype=float)

    metadata = {
        "window_size": window_size,
        "stride": stride,
        "num_windows": len(windows),
        "discarded": discarded,
        "start_indices": start_indices,
    }

    return windows, metadata
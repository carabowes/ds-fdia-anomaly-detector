import numpy as np

"""
Temporal innovation feature computation.

Defines deterministic functions for computing temporal innovations on multivariate measurement vectors,
introducing time-dependent structure designed to expose stealth FDIA behaviour.
"""


def compute_innovations(Z: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    
    # Compute Expontential Moving Average (EMA)-based one-step prediction innovations for multivariate signals.
    
    if Z.ndim != 2:
        raise ValueError("Z must be 2D with shape (T, d)")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1)")

    T, d = Z.shape
    Z_hat = np.zeros((T, d), dtype = float)
    E = np.zeros((T, d), dtype = float)

    Z_hat[0] = Z[0]
    E[0] = 0.0

    for t in range (1, T):
        Z_hat[t] = alpha * Z[t-1] + (1.0 - alpha) * Z_hat[t-1]
        E[t] = Z[t] - Z_hat[t]

    return E
import numpy as np

def wls_estimate(H, z, sigma, reg=1e-6):
    """
    WLS state estimator with diagonal covariance R = sigma^2 I.

    Uses a regularised normal-equations solve to avoid singular-matrix issues:
        (Hᵀ R⁻¹ H + reg I) x = Hᵀ R⁻¹ z
    """
    m, n = H.shape
    R = (sigma ** 2) * np.eye(m)
    R_inv = np.linalg.inv(R)

    Ht_Rinv = H.T @ R_inv
    A = Ht_Rinv @ H
    b = Ht_Rinv @ z

    A_reg = A + reg * np.eye(n)

    x_hat = np.linalg.solve(A_reg, b)
    return x_hat, R



def compute_residuals(H, z, x_hat):
    # Residual vector r = z - H x_hat.
    return z - H @ x_hat


def state_error(x_hat, x_true):
    #State estimation error vector.
    return x_hat - x_true

def run_wls_time_series(Z, H, sigma):
    """
    Run WLS at each timestep.

    Returns:
        R_norms : (T,) residual norms
        X_hat   : (T, n) estimated states
    """
    T = Z.shape[0]
    n = H.shape[1]

    R_norms = np.zeros(T)
    X_hat = np.zeros((T, n))

    for t in range(T):
        # Run WLS using measurements at time t
        x_hat, _ = wls_estimate(H, Z[t], sigma)
        # Compute residual vector
        r = Z[t] - H @ x_hat

        # Store estimated state and residual norm
        X_hat[t] = x_hat
        R_norms[t] = np.linalg.norm(r)

    return R_norms, X_hat

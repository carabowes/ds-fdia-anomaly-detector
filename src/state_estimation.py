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
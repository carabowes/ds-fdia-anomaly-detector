import numpy as np

def standard_FDIA(z, attacked_indices, shift):
    """
    Non stealthy FDI attack: add a fixed shift to selected measurements.
    """
    z_attack = z.copy()
    z_attack[attacked_indices] += shift
    return z_attack

def random_attack(z, attacked_indices, rng, scale):
    """
    Random attack: add random noise (N(0, scale^2)) to selected measurements.
    """
    z_attack = z.copy()
    z_attack[attacked_indices] += rng.normal(0.0, scale, size=len(attacked_indices))
    return z_attack

def stealth_FDIA(H, attacked_indices, alpha, rng):
    """
    Stealth attack: construct a = H c so that residual tests are (nearly) blind.

    c: chosen random direction in state space, scaled by alpha.
    a_full = H c: attack in measurement space.
    We then apply it only on attacked_indices.
    """
    n = H.shape[1]

    # Random direction in state space
    c = rng.standard_normal(n)
    c = c / np.linalg.norm(c)
    c = alpha * c

    a_full = H @ c

    a = np.zeros_like(a_full)
    a[attacked_indices] = a_full[attacked_indices]

    return a

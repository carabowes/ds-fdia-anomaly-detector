import numpy as np
from typing import Tuple

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

# # def stealth_FDIA(H: np.ndarray, z_clean: np.ndarray, attacked_indices: np.ndarray, percent: float, rng: np.random.Generator,):
# def stealth_FDIA(H: np.ndarray, z_clean: np.ndarray, attacked_indices: np.ndarray, percent: float, c_direction: np.ndarray,):
#     """
#     Stealth attack: construct a = H c so that residual tests are (nearly) blind.

#     c: chosen random direction in state space, scaled by alpha.
#     a_full = H c: attack in measurement space.
#     We then apply it only on attacked_indices.
#     """

#     a_full = H @ c_direction

#     a = np.zeros_like(a_full)

#     for idx in attacked_indices:
#         base_mag = abs(z_clean[idx]) + 1e-6
#         target_mag = percent * base_mag

#         if abs(a_full[idx]) > 1e-8:
#             scale = target_mag / abs(a_full[idx])
#             a[idx] = a_full[idx] * scale
#         else:
#             a[idx] = 0.0

#     return a

def stealth_FDIA(
    H: np.ndarray,
    z_clean: np.ndarray,
    percent: float,
    c_direction: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TRUE stealth FDIA: a = alpha * (H @ c_direction)
    where alpha is chosen so that ||a||_2 ≈ percent * ||z_clean||_2.

    Returns:
      z_attacked, a
    """
    z_clean = np.asarray(z_clean, dtype=float).reshape(-1)
    c_direction = np.asarray(c_direction, dtype=float).reshape(-1)

    a_dir = H @ c_direction
    z_norm = float(np.linalg.norm(z_clean))
    a_norm = float(np.linalg.norm(a_dir))

    alpha = float(percent) * z_norm / (a_norm + 1e-12)
    a = alpha * a_dir
    return z_clean + a, a


def make_bus_targeted_c(
    *,
    n_state: int,
    attack_buses: list[int],
    slack_bus: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build c_direction where only targeted buses (excluding slack) have non-zero entries.
    DC assumption: slack removed, so bus b -> state index (b-1) if slack is 0.
    """
    c = np.zeros(n_state, dtype=float)

    for b in attack_buses:
        if b == slack_bus:
            continue  # never target slack
        state_idx = b - 1  # slack=0 mapping
        if 0 <= state_idx < n_state:
            c[state_idx] = rng.standard_normal()

    # Fallback if user accidentally passes only slack bus
    if np.linalg.norm(c) < 1e-12:
        # pick a random non-slack state
        idx = rng.integers(0, n_state)
        c[idx] = 1.0

    c /= (np.linalg.norm(c) + 1e-12)
    return c



    # n = H.shape[1]

    # #1) random direction in state space
    # c = rng.standard_normal(n)
    # c = c / np.linalg.norm(c)
    # a_full = H @ c

    # #2) Scale relative to measurement magnitude (10-15%)
    # a = np.zeros_like(a_full)
    # for idx in attacked_indices:
    #     base_mag = abs(z_clean[idx]) + 1e-6  # avoid zero scaling
    #     target_mag = percent * base_mag

    #     # scale stealth vector entry to bounded magnitude
    #     if abs(a_full[idx]) > 1e-8:
    #         scale = target_mag / abs(a_full[idx])
    #         a[idx] = a_full[idx] * scale
    #     else:
    #         a[idx] = 0.0

    # return a

    # n = H.shape[1]

    # # Random direction in state space
    # c = rng.standard_normal(n)
    # c = c / np.linalg.norm(c)
    # c = alpha * c

    # a_full = H @ c

    # a = np.zeros_like(a_full)
    # a[attacked_indices] = a_full[attacked_indices]

    # return a

def raised_cosine_envelope(t: int, start: int, end: int) -> float:
    """
    Smooth envelope in [0,1] over the interval [start, end).
    - 0 at start
    - 1 around the middle
    - 0 at end (exclusive)
    """
    if t < start or t >= end:
        return 0.0
    # map t in [start, end) to phase in [0, pi]
    phase = np.pi * (t - start) / max(1, (end - start - 1))
    # 0->1->0 smoothly
    return float(np.sin(phase) ** 2)
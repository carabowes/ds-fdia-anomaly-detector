import numpy as np
from typing import List, Tuple, Optional

Episode = Tuple[int, int]

def generate_random_attack(
    T: int,
    rng: np.random.Generator,
    p_start: float = 0.01,
    duration_min: int = 50,
    duration_max: int = 150,
    cooldown: int = 50,
    no_attack_before: int = 0,
) -> List[Episode]:
    
    """
    Generate Probabilistic FDIA episodes over [0, T].
    - At each timestep, with probability p_start, an attack may begin (if not already attacking)
    - Attack duration is sampled uniformly from [duration_min, duration_max].
    - After an attack ends, cooldown timesteps must pass before another can start.
    - No attack may start before no_attack_before.
    - Deterministic given rng.
    """

    if T <= 0:
        return []

    if not (0.0 <= p_start <= 1.0):
        raise ValueError("p_start must be in [0,1].")
    if duration_min <= 0 or duration_max <= 0 or duration_min > duration_max:
        raise ValueError("Invalid duration_min/duration_max.")
    if cooldown < 0:
        raise ValueError("cooldown must be >= 0.")
    if no_attack_before < 0:
        no_attack_before = 0

    episodes: List[Episode] = []
    t = int(no_attack_before)
    last_end = -int(cooldown)

    while t < T:
        can_start = (t - last_end) >= cooldown

        if can_start and rng.random() < p_start:
            duration = int(rng.integers(duration_min, duration_max + 1))
            start = t
            end = min(t + duration, T)
            episodes.append((start, end))
            last_end = end
            t = end  # jump past attack
        else:
            t += 1

    return episodes

    # episodes: List[Tuple[int, int]] = []
    # t = max(no_attack_before, 0)
    # last_end = -cooldown
    # while t < T:
    #     can_start = (t - last_end) >= cooldown

    #     if can_start and rng.random() < p_start:
    #         duration = int(rng.integers(duration_min, duration_max + 1))
    #         start = t
    #         end = min(t + duration, T)

    #         episodes.append((start, end))
    #         last_end = end
    #         t = end  # jump past attack
    #     else:
    #         t += 1

    # return episodes
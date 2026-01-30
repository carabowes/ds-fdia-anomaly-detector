from typing import List, Optional, Sequence, Tuple
import numpy as np
import pandapower as pp
from src.pipeline.simulation import build_dc_measurement_model, simulate_measurements
from src.pipeline.attacks import stealth_FDIA, random_attack, standard_FDIA

def run_time_series(
    net, #pandapower network
    T=200, # number of timesteps
    p_noise_std=0.01, # standard deviation of active power (P) load noise
    q_noise_std=0.0, # standard deviation of reactive power (Q) load noise (0 = disabled)
    meas_noise_std=0.04, # std dev of measurement noise
    rng=None,
    seed=42,
    
):
    if rng is None:
        rng = np.random.default_rng(seed)

    Z = []  # measurement vectors
    X_true = [] # true states
    converged = np.zeros(T, dtype=bool)  # tracks whether PF converged at each timestep

    # Store baseline load values
    base_p = net.load.p_mw.values.copy() #base active power demand at each load bus
    base_q = net.load.q_mvar.values.copy() if q_noise_std > 0 else None #base reactive power demand (only if enabled)

    H = None

    for t in range(T):
        # Load perturbation (drives time evolution)
        p_noise = rng.normal(0.0, p_noise_std, size=len(base_p))
        net.load.loc[:, "p_mw"] = base_p * (1 + p_noise) #apply noise 

        if base_q is not None:
            q_noise = rng.normal(0.0, q_noise_std, size=len(base_q))
            net.load.loc[:, "q_mvar"] = base_q * (1 + q_noise) #apply noise

        # Power flow
        try:
            pp.runpp(
                net,
                algorithm="nr", 
                init="dc",
                calculate_voltage_angles=True,
            )
            converged[t] = True #PF converged successfully
        except Exception:
            converged[t] = False

            # Carry forward last valid state to preserve time-series continuity
            if Z:
                Z.append(Z[-1].copy())
                X_true.append(X_true[-1].copy())
                continue
            else:
                raise RuntimeError("Power flow failed at t=0.")
            
        # Build DC model
        H, x_true, _, _ = build_dc_measurement_model(net)

        # Generate noisy measurements
        z_noisy = simulate_measurements(H, x_true, meas_noise_std, rng)

        # Store results for this timestep
        Z.append(z_noisy)
        X_true.append(x_true)

    return np.array(Z), np.array(X_true), converged, H

# Episode Helpers
Episode = Tuple[int, int]

def normalise_episodes(
    T: int,
    episodes,        #: Optional[Sequence[Episode]],
    start: int,
    end: int,
) -> List[Episode]:
    
    # Normalise attack scheduling input into a validated list of episodes.
    # If episodes is provided use it, else fall back to legacy start/end

    # Build episode list
    if episodes is None:
        episodes_list = [(int(start), int(end))]
    else:
        episodes_list = []
        for ep in episodes:
            if isinstance(ep, dict):
                s = int(ep["start"])
                e = int(ep["end"])
            else:
                s, e = ep
                s = int(s)
                e = int(e)
            episodes_list.append((s, e))

    # Validate and clean episodes
    cleaned = []
    for (s, e) in episodes_list:
        if s < 0 or e < 0:
            raise ValueError(f"Episode has negative bounds: {(s, e)}")
        if s >= e:
            raise ValueError(f"Episode start >= end: {(s, e)}")
        if s >= T:
            continue
        e = min(e, T)
        cleaned.append((s, e))

    cleaned.sort(key=lambda x: x[0])
    return cleaned

def episodes_to_attack_mask(T: int, episodes: Sequence[Episode]) -> np.ndarray:
    # Convert episodes [(s,e), ...] into a binary attack_mask of length T.
    # 1 indicates attack active, 0 indicates no attack.
    mask = np.zeros(T, dtype=int)
    for (s, e) in episodes:
        if e <= s:
            continue
        mask[s:e] = 1
    return mask

def iter_attack_timesteps(attack_mask: np.ndarray) -> np.ndarray:
    #Return sorted timesteps where attack is active. Using np.nonzero preserves increasing order.
    return np.nonzero(attack_mask)[0]

def inject_fdi_time_series(
    Z,
    H,
    attack_type: str ="standard",
    #attack_type="random",
    #attack_type="stealth",
    attacked_indices = None,
    alpha= 0.05, #stealth FDIA magnitude parameter
    start= 50,
    end = 150,
    episodes = None,
    rng = None,
    shift = 0.1,
    scale = 0.05,
    seed = 42,
    random_strength: bool = False,
    strength_min = 0.5,
    strength_max = 1.5,
):
    
    # Inject FDI attacks over a time window
    if rng is None:
        rng = np.random.default_rng(seed)

    T, m = Z.shape
    Z_att = Z.copy() # Copy original measurements
    attack_mask = np.zeros(T, dtype=int) #Binary vector indicating attack presence

    if attacked_indices is None:
        attacked_indices = np.arange(m)

     # Resolve and validate attack schedule
    episodes_list = normalise_episodes(T=T, episodes=episodes, start=start, end=end)
    attack_mask = episodes_to_attack_mask(T=T, episodes=episodes_list)
    episode_log = []
    
     # Apply attacks episode-by-episode (not per-timestep)
    for (s, e) in episodes_list:
            # optional per episode randomisation of attack strength
        if random_strength:
            factor_alpha = float(rng.uniform(strength_min, strength_max))
            factor_shift = float(rng.uniform(strength_min, strength_max))
            factor_scale = float(rng.uniform(strength_min, strength_max))
        else:
            factor_alpha = 1.0
            factor_shift = 1.0
            factor_scale = 1.0

        alpha_ep = alpha * factor_alpha
        shift_ep = shift * factor_shift
        scale_ep = scale * factor_scale

        # Log parameters
        episode_log.append({
            "start": int(s),
            "end": int(e),
            "alpha": float(alpha_ep),
            "shift": float(shift_ep),
            "scale": float(scale_ep),
        })

        for t in range(s, e):
            if attack_type == "standard":
                Z_att[t] = standard_FDIA(Z_att[t], attacked_indices, shift_ep)

            elif attack_type == "random":
                Z_att[t] = random_attack(Z_att[t], attacked_indices, rng, scale_ep)

            elif attack_type == "stealth":
                a = stealth_FDIA(H, attacked_indices, alpha_ep, rng)
                Z_att[t] += a

            else:
                raise ValueError(f"Unknown attack_type: {attack_type}")

    return Z_att, attack_mask, episode_log
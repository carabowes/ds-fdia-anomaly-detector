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

def inject_fdi_time_series(
    Z,
    H,
    attack_type="standard",
    #attack_type="random",
    #attack_type="stealth",
    attacked_indices=None,
    alpha=0.05, #stealth FDIA magnitude parameter
    start=50,
    end=150,
    rng=None,
    shift=0.1,
    scale=0.05,
    seed =42,
):
    
    # Inject FDI attacks over a time window
    if rng is None:
        rng = np.random.default_rng(seed)

    T, m = Z.shape
    Z_att = Z.copy() # Copy original measurements
    attack_mask = np.zeros(T, dtype=int) #Binary vector indicating attack presence

    if attacked_indices is None:
        attacked_indices = np.arange(m)

    # Inject attacks only within specified time window
    for t in range(start, end):
        attack_mask[t] = 1

        if attack_type == "standard":
            Z_att[t] = standard_FDIA(Z_att[t], attacked_indices, shift)

        elif attack_type == "random":
            Z_att[t] = random_attack(Z_att[t], attacked_indices, rng, scale)

        elif attack_type == "stealth":
            a = stealth_FDIA(H, attacked_indices, alpha, rng)
            Z_att[t] += a
        else:
            raise ValueError(f"Unknown attack_type: {attack_type}")

    return Z_att, attack_mask


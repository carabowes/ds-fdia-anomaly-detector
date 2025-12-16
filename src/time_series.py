import numpy as np
import pandapower as pp
from src.simulation import build_dc_measurement_model, simulate_measurements
from src.attacks import stealth_FDIA, random_attack, standard_FDIA

def run_time_series(
    net, #pandapower network
    T=200, # number of timesteps
    load_noise_std=0.01, # std dev of load noise
    meas_noise_std=0.04, # std dev of measurement noise
    rng=None
):
    
    """
    Generate a continuous time series of DC measurements.
    Returns:
        Z       : (T, m) measurement time series
        X_true  : (T, n) true state (bus angle) time series
    """

    if rng is None:
        rng = np.random.default_rng(42)

    Z = [] 
    X_true = [] # true states

    # Store base active power demand at each load bus
    base_p = net.load.p_mw.values.copy()

    for t in range(T):
        # Disturb loads with random noise
        noise = rng.normal(0.0, load_noise_std, size=len(base_p))
        net.load.loc[:, "p_mw"] = base_p * (1 + noise)

        # Run power flow
        pp.runpp(
            net,
            algorithm="nr", 
            init="dc", 
            calculate_voltage_angles=True
        )

        # Build DC model
        H, x_true, z_true, _ = build_dc_measurement_model(net)

        # Generate noisy measurements
        z_no = simulate_measurements(H, x_true, meas_noise_std, rng)

        # Store results for this timestep
        Z.append(z_no)
        X_true.append(x_true)

    return np.array(Z), np.array(X_true)

def inject_fdi_time_series(
    Z,
    H,
    attack_type="stealth",
    attacked_indices=None,
    alpha=0.05,
    start=50,
    end=150,
    rng=None,
    shift=0.1,
    scale=0.05
):
    
    # Inject FDI attacks over a time window
    if rng is None:
        rng = np.random.default_rng(42)

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

import numpy as np
import pandapower as pp
import pandapower.networks as pn

def load_ieee14():
    net = pn.case14()
    pp.runpp(net, algorithm="nr", init="dc", calculate_voltage_angles=True)
    return net

def build_dc_measurement_model(net):
    """
    Build a DC state estimation model using branch-flow measurements:

        z_k = (theta_i - theta_j) / x_ij

    Returns:
        H      : measurement matrix (m x n)
        x_true : true state vector (non-slack angles)
        z_true : true flow measurements (m,)
        mask   : boolean mask for non-slack buses
    """
    # 1. Identify slack and true angles
    
    slack_bus = net.ext_grid.bus.values[0]
    va_deg = net.res_bus.va_degree.values
    va_rad = np.deg2rad(va_deg)

    nb = len(net.bus)

    # Mask of non-slack buses
    mask = np.ones(nb, dtype=bool)
    mask[slack_bus] = False
    non_slack_buses = np.where(mask)[0]
    n = len(non_slack_buses)

    # True state vector (angles at non-slack buses)
    x_true = va_rad[mask]

    #Build measurement matrix H
    rows = []
    z_true_list = []

    for _, line in net.line.iterrows():
        i = int(line.from_bus)
        j = int(line.to_bus)

        # Determine reactance x
        if 'x_ohm_per_km' in line.index:
            x = line.x_ohm_per_km * line.length_km
        elif 'x_ohm' in line.index:
            x = line.x_ohm
        elif 'x_pu' in line.index:
            x = line.x_pu
        elif 'x' in line.index:
            x = line.x
        else:
            raise ValueError("Could not determine reactance column.")

        row = np.zeros(n)

        # Only fill entries for non-slack buses
        if i in non_slack_buses:
            idx_i = np.where(non_slack_buses == i)[0][0]
            row[idx_i] =  1.0 / x

        if j in non_slack_buses:
            idx_j = np.where(non_slack_buses == j)[0][0]
            row[idx_j] = -1.0 / x

        rows.append(row)

        # True flow
        theta_i = va_rad[i]
        theta_j = va_rad[j]
        z_true_list.append((theta_i - theta_j) / x)

    H = np.vstack(rows)
    z_true = np.array(z_true_list)

    print("H shape =", H.shape)
    print("x_true shape =", x_true.shape)
    print("z_true shape =", z_true.shape)

    return H, x_true, z_true, mask

def simulate_measurements(H, x_true, sigma, rng):
    """
    Simulate noisy measurements z = H x_true + e,
    where e ~ N(0, sigma^2 I).
    """
    m = H.shape[0]
    noise = rng.normal(loc=0.0, scale=sigma, size=m)
    z = H @ x_true + noise
    return z

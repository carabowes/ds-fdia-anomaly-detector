import numpy as np
import pandapower as pp
import pandapower.networks as pn

"""
Time-series power-system simulation utilities.

Implements load perturbations, power-flow execution, and measurement generation over time for the IEEE test networks. 
Provides the physical dynamics that drive the FDIA experiments.
"""

def load_test_case(case="case14"):
    if case == "case14":
        net = pn.case14()
    elif case == "case9":
        net = pn.case9()
    else:
        raise ValueError(f"Unknown test case: {case}")

    pp.runpp(net, algorithm="nr", init="dc", calculate_voltage_angles=True)
    return net

def build_dc_measurement_model(net):

    slack_bus = net.ext_grid.bus.values[0]

    va_deg = net.res_bus.va_degree.values
    va_rad = np.deg2rad(va_deg)

    nb = len(net.bus)

    mask = np.ones(nb, dtype=bool)
    mask[slack_bus] = False

    non_slack_buses = np.where(mask)[0]
    n = len(non_slack_buses)

    # map bus -> state index
    bus_to_idx = {bus: i for i, bus in enumerate(non_slack_buses)}

    x_true = va_rad[mask]

    rows = []
    z_true_list = []
    line_row_count = 0
    inj_row_count = 0
    # -----------------------
    # Line flow measurements
    # -----------------------
# -----------------------
# Line flow measurements
# -----------------------
    for _, line in net.line.iterrows():

        i = int(line.from_bus)
        j = int(line.to_bus)

        # convert reactance to per-unit
        x_ohm = line.x_ohm_per_km * line.length_km
        base_kv = net.bus.vn_kv.iloc[i]
        Z_base = (base_kv ** 2) / net.sn_mva
        x = x_ohm / Z_base

        row = np.zeros(n)

        if i in bus_to_idx:
            row[bus_to_idx[i]] = 1.0 / x

        if j in bus_to_idx:
            row[bus_to_idx[j]] = -1.0 / x

        rows.append(row)
        line_row_count += 1
        theta_i = va_rad[i]
        theta_j = va_rad[j]

        z_true_list.append((theta_i - theta_j) / x)


    # -----------------------
    # Bus injection measurements
    # -----------------------
    for bus in non_slack_buses:

        row = np.zeros(n)

        for _, line in net.line.iterrows():

            i = int(line.from_bus)
            j = int(line.to_bus)

            # convert reactance to per-unit
            x_ohm = line.x_ohm_per_km * line.length_km
            base_kv = net.bus.vn_kv.iloc[i]
            Z_base = (base_kv ** 2) / net.sn_mva
            x = x_ohm / Z_base

            if i == bus or j == bus:

                other = j if i == bus else i

                row[bus_to_idx[bus]] += 1.0 / x

                if other in bus_to_idx:
                    row[bus_to_idx[other]] -= 1.0 / x

        rows.append(row)
        inj_row_count += 1

        theta_i = va_rad[bus]

        P_i = 0.0

        for _, line in net.line.iterrows():

            i = int(line.from_bus)
            j = int(line.to_bus)

            # convert reactance to per-unit
            x_ohm = line.x_ohm_per_km * line.length_km
            base_kv = net.bus.vn_kv.iloc[i]
            Z_base = (base_kv ** 2) / net.sn_mva
            x = x_ohm / Z_base

            if i == bus:
                P_i += (theta_i - va_rad[j]) / x
            elif j == bus:
                P_i += (theta_i - va_rad[i]) / x

    z_true_list.append(P_i)

    H = np.vstack(rows)
    z_true = np.array(z_true_list)
    # print("num lines in net.line:", len(net.line))
    # print("non_slack_buses:", non_slack_buses)
    # print("line rows added:", line_row_count)
    # print("injection rows added:", inj_row_count)
    # print("total rows added:", len(rows))

    return H, x_true, z_true, mask

def simulate_measurements(H, x_true, sigma, rng):
    """
    Simulate noisy measurements z = H x_true + e,
    where e ~ N(0, sigma^2 I).
    """
    m = H.shape[0]
    noise = rng.normal(loc=0.0, scale=sigma, size=m)
    z = H @ x_true + noise
    # print("H shape:", H.shape)
    # print("max |H|:", np.max(np.abs(H)))
    # print("min nonzero |H|:", np.min(np.abs(H[np.nonzero(H)])))

    return z

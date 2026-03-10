import pandapower as pp
import numpy as np

# def apply_control(net, u_t, ramp_limits):
#     if u_t is None:
#         return

#     for i, p_new in enumerate(u_t["gen_p"]):
#         p_old = net.gen.at[i, "p_mw"]
#         ramp = ramp_limits.get(i, 10.0)  # MW per step

#         p_applied = np.clip(
#             p_new,
#             p_old - ramp,
#             p_old + ramp,
#         )

#         net.gen.at[i, "p_mw"] = p_applied


def apply_control(net, u_t, ramp_limits=None):
    """
    Apply redispatch with:
      1) ramp-rate limit per timestep
      2) generator min/max bounds (if present)
    """
    if u_t is None:
        return

    if ramp_limits is None:
        ramp_limits = {}  # IMPORTANT: prevent None bugs

    for i, p_target in enumerate(u_t["gen_p"]):
        if i not in net.gen.index:
            continue

        p_old = float(net.gen.at[i, "p_mw"])
        ramp = float(ramp_limits.get(int(i), 10.0))  # MW per step

        # --- 1) ramp clamp ---
        p_ramped = float(np.clip(p_target, p_old - ramp, p_old + ramp))

        # --- 2) min/max clamp (if available) ---
        if "min_p_mw" in net.gen.columns:
            p_min = float(net.gen.at[i, "min_p_mw"])
        else:
            p_min = -np.inf

        if "max_p_mw" in net.gen.columns:
            p_max = float(net.gen.at[i, "max_p_mw"])
        else:
            p_max = np.inf

        p_applied = float(np.clip(p_ramped, p_min, p_max))

        net.gen.at[i, "p_mw"] = p_applied

def ensure_gen_limits(net, default_headroom_mw: float = 50.0):
    """
    Ensure pandapower net.gen has min/max bounds.
    """
    if len(net.gen) == 0:
        return

    if "min_p_mw" not in net.gen.columns:
        net.gen["min_p_mw"] = np.nan
    if "max_p_mw" not in net.gen.columns:
        net.gen["max_p_mw"] = np.nan

    for i in net.gen.index:
        p0 = float(net.gen.at[i, "p_mw"])

        # If missing or NaN, set bounds around current p_mw
        if not np.isfinite(net.gen.at[i, "min_p_mw"]):
            net.gen.at[i, "min_p_mw"] = p0 - default_headroom_mw
        if not np.isfinite(net.gen.at[i, "max_p_mw"]):
            net.gen.at[i, "max_p_mw"] = p0 + default_headroom_mw

        # Safety: enforce min <= max
        if net.gen.at[i, "min_p_mw"] > net.gen.at[i, "max_p_mw"]:
            net.gen.at[i, "min_p_mw"], net.gen.at[i, "max_p_mw"] = (
                net.gen.at[i, "max_p_mw"],
                net.gen.at[i, "min_p_mw"],
            )
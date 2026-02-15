import pandapower as pp
import numpy as np

def apply_control(net, u_t, ramp_limits):
    if u_t is None:
        return

    for i, p_new in enumerate(u_t["gen_p"]):
        p_old = net.gen.at[i, "p_mw"]
        ramp = ramp_limits.get(i, 10.0)  # MW per step

        p_applied = np.clip(
            p_new,
            p_old - ramp,
            p_old + ramp,
        )

        net.gen.at[i, "p_mw"] = p_applied